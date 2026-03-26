import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

from experta import *
import sqlite3


class Laptop(Fact):
    pass


class Recommendation(Fact):
    pass


class LaptopAdvisor(KnowledgeEngine):
    @Rule(Laptop(budget=P(lambda x: x < 50000), purpose="учёба"))
    def rule1(self):
        self.declare(Recommendation(parameter="processor", value="Intel Core i3 / AMD Ryzen 3"))

    @Rule(Laptop(purpose="игры", budget=P(lambda x: x > 80000)))
    def rule2(self):
        self.declare(Recommendation(parameter="graphics", value="NVIDIA RTX"))

    @Rule(Laptop(weight=P(lambda x: x < 1.5), screen_size=P(lambda x: x <= 14)))
    def rule3(self):
        self.declare(Recommendation(parameter="type", value="ультрабук"))


def init_db():
    conn = sqlite3.connect("laptop_advisor.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS laptops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            budget INTEGER,
            purpose TEXT,
            weight REAL,
            screen_size REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            laptop_id INTEGER,
            parameter TEXT,
            value TEXT,
            FOREIGN KEY (laptop_id) REFERENCES laptops(id)
        )
    """)

    conn.commit()
    return conn, cursor


def save_laptop(cursor, budget, purpose, weight, screen_size):
    cursor.execute("""
        INSERT INTO laptops (budget, purpose, weight, screen_size)
        VALUES (?, ?, ?, ?)
    """, (budget, purpose, weight, screen_size))
    return cursor.lastrowid


def save_recommendation(cursor, laptop_id, parameter, value):
    cursor.execute("""
        INSERT INTO recommendations (laptop_id, parameter, value)
        VALUES (?, ?, ?)
    """, (laptop_id, parameter, value))


# Инициализация БД
conn, cursor = init_db()

# Данные ноутбука
budget = 90000
purpose = "игры"
weight = 3
screen_size = 15

# Сохранение ноутбука в БД
laptop_id = save_laptop(cursor, budget, purpose, weight, screen_size)

# Инициализация экспертной системы
engine = LaptopAdvisor()
engine.reset()

# Добавление факта
engine.declare(Laptop(
    budget=budget,
    purpose=purpose,
    weight=weight,
    screen_size=screen_size
))

# Запуск вывода
engine.run()

# Сохранение рекомендаций в БД
print("\nНайденные рекомендации:")
for fact in engine.facts.values():
    if isinstance(fact, Recommendation):
        parameter = fact["parameter"]
        value = fact["value"]
        print(f"{parameter}: {value}")
        save_recommendation(cursor, laptop_id, parameter, value)

conn.commit()

# Просмотр сохранённых данных
print("\nНоутбуки в базе:")
cursor.execute("SELECT * FROM laptops")
for row in cursor.fetchall():
    print(row)

print("\nРекомендации в базе:")
cursor.execute("""
    SELECT recommendations.id, recommendations.laptop_id, recommendations.parameter, recommendations.value
    FROM recommendations
""")
for row in cursor.fetchall():
    print(row)

conn.close()