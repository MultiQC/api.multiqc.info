from datetime import datetime
from typing import Optional
from fastapi import BackgroundTasks, FastAPI
from sqlmodel import Field, Session, SQLModel, create_engine, select
from os import getenv
import time

app = FastAPI()

sql_url = getenv("DATABASE_URL")
engine = create_engine(sql_url, echo=True)


class Visit(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    version_multiqc: str
    version_python: str
    operating_system: str
    installation_method: str
    called_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def create_visits():
    visit_1 = Visit(
        version_multiqc="v0.7",
        version_python="2.7",
        operating_system="windows",
        installation_method="conda",
    )
    visit_2 = Visit(
        version_multiqc="v1.13",
        version_python="3.8",
        operating_system="osx",
        installation_method="docker",
    )
    visit_3 = Visit(
        version_multiqc="v1.15",
        version_python="3.11.1",
        operating_system="linux",
        installation_method="singularity",
    )

    with Session(engine) as session:
        session.add(visit_1)
        session.add(visit_2)
        session.add(visit_3)

        session.commit()


def select_visits():
    with Session(engine) as session:
        statement = select(Visit)
        results = session.exec(statement)
        for hero in results:
            print(hero)


def run_db_tasks():
    print("very slow background task started")
    time.sleep(5)
    create_db_and_tables()
    create_visits()
    select_visits()
    print("background task finished")


@app.get("/")
async def root(background_tasks: BackgroundTasks):
    return {"message": "Hello World", "available_endpoints": ["/version"]}


@app.get("/version")
async def root(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_db_tasks)
    return {
        "latest_release": "v.14",
        "broadcast_message": "",
        "latest_release_date": "2023-01-23",
        "module_warnings": [],
    }
