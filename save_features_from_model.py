from bert_nli import BertNLIModel
from utils.nli_data_reader import NLIDataReader
import os
import pickle
import json
import tqdm
from torch.nn import Module
import torch

features_out = []

def cp_register_hook(model:Module, layer_names:list):

    def forward_hook(module, input, output):
        feature  = torch.mean(output,dim=1).squeeze().cpu().detach().numpy()
        features_out.append(feature)
        return None

    hooks = []
    for name, module in model.named_modules():
            # print(name)
        if name in layer_names:
            # print("find layer ", name)
            hooks.append(module.register_forward_hook(forward_hook))

    return hooks

def construct_mqnli_concept_dataset(
    data_list,
    sample_num,
    explained_model,
    target_layer_names,
    dataset_file_name
    ):

    """ 
    Return:
        concept_dataset_dict: is a dict in format 
            {
                'concept name 1': 
                (
                    [original model input numpy],
                    [features numpy for target layer],
                    [concept annotation value]
                ),
                'concept name 1': (...),
                ...
            }
    """

    concept_dataset = {}

    for item in data_list:
        concept_label = item['concept_label']
        for concept_key in concept_label.keys():
            if concept_key in concept_dataset:
                if concept_dataset[concept_key]["concept_label"].count(concept_label[concept_key])<sample_num:
                    concept_dataset[concept_key]["input"].append(item['input'])
                    concept_dataset[concept_key]["concept_label"].append(concept_label[concept_key])
            else:
                concept_dataset[concept_key] = {
                    "input": [item['input']],
                    "concept_label": [concept_label[concept_key]]
                }

    cp_register_hook(explained_model, target_layer_names)

    for c in concept_dataset.keys():
        print(c,len(concept_dataset[c]["concept_label"]))
        a = {}
        for i in concept_dataset[c]["concept_label"]:
            a[i] = concept_dataset[c]["concept_label"].count(i)
        print(a)

        concept_dataset[c]["concept_features"] = {}

        with tqdm.tqdm(total=len(concept_dataset[c]["concept_label"])) as tbar:
            for input_sample in concept_dataset[c]["input"]:
                labels, probs = explained_model([input_sample])

                for idx, layer_name in enumerate(target_layer_names):
                    if layer_name in concept_dataset[c]["concept_features"]:
                        concept_dataset[c]["concept_features"][layer].append(features_out[idx])
                    else:
                        concept_dataset[c]["concept_features"][layer] = [features_out[idx]]

                features_out.clear()

                tbar.update(1)

    pickle.dump(concept_dataset, open(dataset_file_name, 'wb'))


if __name__ == "__main__":
    ratio = "0.5"
    bert_type = "bert-large"
    # bert_type = 'albert-base-v2'
    mpath = f"output\\{ratio}\\mqnli_bert-large-2024-02-04_12-07-02\\nli_model_acc0.9997999799979999.state_dict"

    sample_num = 200
    dataset_folder="datasets\\mqnli_concept"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    model = BertNLIModel(model_path=mpath, bert_type=bert_type)
    print(model)

    # Read the dataset
    test_data = json.load(open('datasets\\mqnli_causal\\0.5gendata.test.json','r'))

    # print(len(test_data),test_data)

    target_layer_names = []

    for name, layer in model.named_modules():
        if ("albert_layer_groups.0.albert_layers.0.ffn_output" in name):
            if name not in target_layer_names:
                target_layer_names.append(name)

        if ("output.LayerNorm" in name) and ("attention" not in name):
            if name not in target_layer_names:
                target_layer_names.append(name)
    
    print(target_layer_names)


    dataset_file_name = os.path.join(dataset_folder, f"mqnli_concept_{sample_num}_{bert_type}_{ratio}.txt")

    construct_mqnli_concept_dataset(
        test_data,
        sample_num,
        model,
        target_layer_names,
        dataset_file_name)
