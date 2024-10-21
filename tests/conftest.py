import pytest

def pytest_collection_modifyitems(session, config, items):
    for item in items:
        if "unit_tests" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "functional_tests" in str(item.fspath):
            item.add_marker(pytest.mark.functional)