import json

def read_json(filename):    
    with open(filename, 'r') as openfile:
        return json.load(openfile)

def write_json(to_json, filename):
    json_object = json.dumps(to_json)

    with open(filename, "w") as outfile:
        outfile.write(json_object)