import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--bigmem", 
        action="store_true", 
        default=False, 
        help="run tests that require a large amount of memory"
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", 
        "bigmem: mark test as requiring a large amount of memory"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--bigmem"):
        skip_bigmem = pytest.mark.skip(reason="need --bigmem option to run")
        for item in items:
            if "bigmem" in item.keywords:
                item.add_marker(skip_bigmem)
