import json


def parse_json2str(json_file):
    str2json = {}
    for key in json_file:
        str2json[key] = str(json_file[key])
    return str2json


def parse_str2json(json_file):
    unstr2json = {}
    for key in json_file:
        string = json_file[key]
        if string.startswith(("{", "[")):
            unstr2json[key] = eval(json_file[key])
        else:
            unstr2json[key] = json_file[key]
    return unstr2json


def parse_json4kaapana(json_file):
    myDict = {}
    myDict["predicted_bodypart_string"] = json_file["body part examined tag"]
    myDict["prediction_parameters_string"] = json_file

    return parse_json2str(myDict)


def test_parser():
    myDict = {
        "a": [1, 2, 3, 4],
        "b": "Hi",
        "c": {"l1": 1, "l2": 2, "l3": 3},
        "d": {"l1": {"a": 0, "b": 0}},
    }

    j1 = parse_json2str(myDict)
    print(j1)

    j2 = parse_str2json(j1)
    print(j2)
    print(j2 == myDict)


if __name__ == "__main__":
    test_parser()
