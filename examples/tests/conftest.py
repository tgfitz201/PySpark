import pytest, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def db_path():
    return "/Users/Fitzgerald/PycharmProjects/PySpark/trades.db"

@pytest.fixture(scope="session")
def repo(db_path):
    from db.repository import TradeRepository
    r = TradeRepository(db_path)
    yield r
    r.close()

@pytest.fixture(scope="session")
def all_trades(repo):
    return repo.list_all()

@pytest.fixture(scope="session")
def results_df(repo):
    return repo.get_results_df(run_id="RUN-1")

@pytest.fixture(scope="session")
def trades_json_path():
    return "/Users/Fitzgerald/PycharmProjects/PySpark/trades.json"

@pytest.fixture(scope="session")
def results_json_path():
    return "/Users/Fitzgerald/PycharmProjects/PySpark/results.json"
