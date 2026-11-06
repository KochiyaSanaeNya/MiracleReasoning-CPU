import os

import pytest

from miracle_reasoning import OpenCLBackend, OpenCLError


@pytest.fixture(scope="session")
def backend():
    # Production defaults to CPU. GPU validation is always an explicit test opt-in.
    try:
        return OpenCLBackend(device_type=os.environ.get("MIRACLE_TEST_DEVICE_TYPE", "cpu"),
                             platform_index=int(os.environ["MIRACLE_TEST_PLATFORM"]) if "MIRACLE_TEST_PLATFORM" in os.environ else None)
    except OpenCLError as exc:
        if "No matching" in str(exc) or "not found" in str(exc):
            pytest.skip(str(exc))
        raise
