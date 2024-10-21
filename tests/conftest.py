import pytest

def pytest_collection_modifyitems(session, config, items):
    sphere_tests = [item for item in items if "sphere_tests" in str(item.fspath)]
    ufuncs_tests = [item for item in items if "ufuncs_tests" in str(item.fspath)]
    functional_tests = [item for item in items if "functional_tests" in str(item.fspath)]

    # Reorder the items list so that unit tests come first
    items[:] = sphere_tests + ufuncs_tests + functional_tests