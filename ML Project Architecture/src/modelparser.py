import json
from Dataclasses.layer import Layer

class ModelParser:
    def __init__(self, path_to_config:str):
        self.config_path = path_to_config
        self.layer_list = list()
        self.hp = list()
        self._parse_config()
        

    def _parse_config(self):
        try:
            with open(self.config_path, "r") as f:
                arch_dic = json.loads(f.read()) # Parse json string  and convert into python dictionery
        except Exception as e:
            print(e)

        for layer_config in arch_dic["Model_config"]:
            Layer_obj = Layer(**layer_config)
            # Layer_obj.print_info()
            self.layer_list.append(Layer_obj)
        
        self.hp.append(arch_dic["Hyperparameters"])

    def get_list(self):
        return self.layer_list
    
    def get_hp(self):
        return self.hp