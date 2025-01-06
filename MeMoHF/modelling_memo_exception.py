class MeMoException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__("MeMo Error: " + self.message)