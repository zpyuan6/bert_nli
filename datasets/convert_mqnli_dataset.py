import json
import os

if __name__ == "__main__":
    for root, folders, files in os.walk("MQNLI"):
        for file in files:
            if 'json' in file:
                continue
            print(file)
            original_data = open(os.path.join(root,file), 'r')

            new_list = []
            json_name = file.replace(".","-")+".json"
            for line in original_data.readlines():
                input_json = json.loads(line)
                input_json["input"] = (input_json["sentence1"].replace("emptystring ","")+" .",input_json["sentence2"].replace("emptystring ","")+" .")
                new_list.append(input_json)

            json_object = json.dumps(new_list)

            with open(os.path.join(root, json_name),'w') as output_file:
                output_file.write(json_object)



