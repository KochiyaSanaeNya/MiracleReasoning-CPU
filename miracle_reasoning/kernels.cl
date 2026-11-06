// OpenCL C 1.2. No subgroups, vendor extensions, FP16, or fixed workgroup size.
// All offsets use size_t so matrices may cross the signed 32-bit byte boundary.
#pragma OPENCL FP_CONTRACT OFF

__kernel void linear(__global const float *x, __global const float *w,
                     __global float *y, int rows, int out_dim, int in_dim) {
    size_t o = get_global_id(0), r = get_global_id(1);
    if (o >= out_dim || r >= rows) return;
    size_t xb = r * in_dim, wb = o * in_dim;
    float4 sum = (float4)(0.0f);
    int j = 0;
    for (; j + 3 < in_dim; j += 4)
        sum += vload4(0, x + xb + j) * vload4(0, w + wb + j);
    float v = sum.x + sum.y + sum.z + sum.w;
    for (; j < in_dim; ++j) v += x[xb+j] * w[wb+j];
    y[r * out_dim + o] = v;
}

__kernel void rms(__global const float *x, __global const float *w,
                  __global float *y, int rows, int width, float eps, int affine) {
    size_t r = get_global_id(0);
    if (r >= rows) return;
    float v = 0.0f;
    for (int j=0; j<width; ++j) { float a=x[r*width+j]; v += a*a; }
    float scale = 1.0f / sqrt(v / (float)width + eps);
    for (int j=0; j<width; ++j)
        y[r*width+j] = x[r*width+j] * scale * (affine ? w[j] : 1.0f);
}

__kernel void rope(__global const float *x, __global const float *freq,
                   __global const float *pos, __global float *y,
                   int rows, int heads, int dim, int split_half) {
    size_t i = get_global_id(0);
    int half_dim = dim/2;
    if (i >= (size_t)rows*heads*half_dim) return;
    int p = i % half_dim;
    size_t head_row = i / half_dim, base = head_row * dim;
    float a = pos[head_row/heads] * freq[p];
    float c = cos(a), s = sin(a);
    int j = split_half ? p : 2*p;
    int k = split_half ? p+half_dim : 2*p+1;
    float u=x[base+j], v=x[base+k];
    y[base+j]=u*c-v*s; y[base+k]=u*s+v*c;
}

__kernel void swiglu(__global const float *g, __global const float *u,
                     __global float *y, int n) {
    size_t i=get_global_id(0);
    if (i<n) y[i] = (g[i]/(1.0f+exp(-g[i]))) * u[i];
}

__kernel void axpy(__global const float *x, __global const float *y,
                   __global float *z, int n, float scale) {
    size_t i=get_global_id(0);
    if (i<n) z[i] = x[i] + scale*y[i];
}

// Logical order of a ring is [start, start+length). Relays follow exact keys.
__kernel void scores(__global const float *q, __global const float *k,
                     __global const float *extra_k, __global float *s,
                     int heads, int kv_heads, int dim, int capacity,
                     int start, int length, int extra_length) {
    size_t t=get_global_id(0), h=get_global_id(1);
    int total=length+extra_length;
    if (t>=total || h>=heads) return;
    int kh=h/(heads/kv_heads);
    size_t kb = t<length ? ((start+t)%capacity)*kv_heads*dim+kh*dim
                        : (t-length)*kv_heads*dim+kh*dim;
    float v=0.0f;
    for (int j=0; j<dim; ++j)
        v += q[h*dim+j] * (t<length ? k[kb+j] : extra_k[kb+j]);
    s[h*total+t] = v / sqrt((float)dim);
}

__kernel void softmax(__global const float *x, __global float *y,
                      int rows, int width) {
    size_t r=get_global_id(0);
    if (r>=rows) return;
    float m=-INFINITY;
    for (int j=0; j<width; ++j) m=fmax(m,x[r*width+j]);
    float denom=0.0f;
    for (int j=0; j<width; ++j) denom += exp(x[r*width+j]-m);
    for (int j=0; j<width; ++j) y[r*width+j]=exp(x[r*width+j]-m)/denom;
}

__kernel void latent_mask(__global float *scores, __global const float *positions,
                          int heads, int length, int memory, int earliest) {
    size_t t=get_global_id(0), h=get_global_id(1);
    if (t<length && h<heads && t>=memory && positions[t]<earliest)
        scores[h*length+t]=-INFINITY;
}

__kernel void mix(__global const float *p, __global const float *v,
                  __global const float *extra_v, __global float *y,
                  int heads, int kv_heads, int dim, int capacity,
                  int start, int length, int extra_length) {
    size_t i=get_global_id(0);
    if (i>=(size_t)heads*dim) return;
    int h=i/dim, j=i%dim, kh=h/(heads/kv_heads), total=length+extra_length;
    float acc=0.0f;
    for (int t=0; t<total; ++t) {
        size_t vb=t<length ? ((start+t)%capacity)*kv_heads*dim+kh*dim+j
                          : (t-length)*kv_heads*dim+kh*dim+j;
        acc += p[h*total+t] * (t<length ? v[vb] : extra_v[vb]);
    }
    y[i]=acc;
}

// Learned CBR queries are [slots, hidden], U is [tokens, hidden].
__kernel void pool_scores(__global const float *u, __global const float *q,
                          __global float *s, int tokens, int slots, int width) {
    size_t t=get_global_id(0), m=get_global_id(1);
    if (t>=tokens || m>=slots) return;
    float v=0.0f;
    for (int j=0; j<width; ++j) v += u[t*width+j]*q[m*width+j];
    s[m*tokens+t]=v/sqrt((float)width);
}

__kernel void pool_mix(__global const float *u, __global const float *p,
                       __global float *z, __global float *anchors,
                       int tokens, int slots, int width, int start_position) {
    size_t j=get_global_id(0), m=get_global_id(1);
    if (j>=width || m>=slots) return;
    float v=0.0f, pos=0.0f;
    for (int t=0; t<tokens; ++t) {
        float w=p[m*tokens+t]; v+=w*u[t*width+j]; pos+=w*(start_position+t);
    }
    z[m*width+j]=v;
    if (j==0) anchors[m]=rint(pos);
}

// Weighted pooling used by FlashLighting's latent cache compaction.
__kernel void merge_rows(__global const float *x, __global float *y,
                         int width, float a, float b) {
    size_t j=get_global_id(0);
    if (j<width) y[j]=(a*x[j]+b*x[width+j])/(a+b);
}

// Block means for Heavy Medium; the final block includes its current prefix.
__kernel void block_means(__global const float *x, __global float *y,
                          int length, int width, int block) {
    size_t j=get_global_id(0), b=get_global_id(1);
    int start=b*block, stop=min(start+block,length);
    if (j>=width || start>=length) return;
    float v=0.0f;
    for (int t=start;t<stop;++t) v+=x[(size_t)t*width+j];
    y[b*width+j]=v/(float)(stop-start);
}
