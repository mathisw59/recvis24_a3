class Config:
    def __init__(self):
        self.a = 1
        self.b = 2

class MyClass:
    def __init__(self, config) -> None:
        self.config = config
        self.a = 3
        print(self.a)

    def __getattr__(self, name):
        print('Returning from config', name)
        return getattr(self.config, name)
    
config = Config()
my_class = MyClass(config)
