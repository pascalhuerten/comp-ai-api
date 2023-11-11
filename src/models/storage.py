from flask import current_app
import psycopg2
import os

class storage:
    def __init__(self):
        self.conn = self._connect_to_db()
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        self._initial_setup()
        current_app.logger.info("Database initialized.")

    def _initial_setup(self):
        current_app.logger.info("Performing initial database setup...")
        # create tables if not exists
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS courses (id bigint PRIMARY KEY, text TEXT NOT NULL)"
        )
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS skills (id VARCHAR(255) PRIMARY KEY, name VARCHAR(255) NOT NULL, taxonomy VARCHAR(255) NOT NULL)"
        )
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS course_skills (id SERIAL PRIMARY KEY, course_id bigint NOT NULL, skill_id VARCHAR(255) NOT NULL, valid BOOLEAN NOT NULL)"
        )
        current_app.logger.info("Database setup complete")

    def _connect_to_db(self):
        # Connect to db
        if not os.getenv('POSTGRES_PASSWORD'):
            raise Exception("POSTGRES_PASSWORD not set")
        conn = psycopg2.connect(f"postgres://postgres:{os.getenv('POSTGRES_PASSWORD')}@postgres/postgres?sslmode=disable")
        current_app.logger.info("Connected to database.")
        return conn