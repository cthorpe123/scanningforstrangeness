
class ConfigLoader:
    def __init__(self, file_):
        with open(file_, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                key, value = line.split("=", 1)
                setattr(self, key.strip(), self._convert_value(value.strip()))
            except ValueError:
                pass

    def _convert_value(self, value):
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            pass
        if value.lower() in {'true', 'false'}:
            return value.lower() == 'true'
        return value 