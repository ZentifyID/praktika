from experta import *
import sqlite3

class Laptop(Fact):
    pass

class Recommendation(Fact):
    pass

class LaptopAdvisor(KnowledgeEngine):
    @Rule(Laptop(budget=P(lambda x: x < 50000), purpose="учёба"))
    def rule1(self):
        self.declare(Recommendation(processor="Intel Core i3 / AMD Ryzen 3"))

    @Rule(Laptop(purpose="игры", budget=P(lambda x: x > 80000)))
    def rule2(self):
        self.decalre(Recommendatio(graphics="NVIDIA RTX"))

    @Rule(Laptop(weight=P(lambda x: x < 1.5), screen_size=P(lambda x: x <= 14)))
    def rule3(self):
        self.declare(Recommendation(type="ультрабук"))

# Инициализация системы
engine = LaptopAdvisor()
engine.reset()

# Добавление фактов
engine.declare(Laptop(budget=45000, purpose="учёба", weight=1.2, screen_size=13.3))

# Запуск вывода
engine.run()