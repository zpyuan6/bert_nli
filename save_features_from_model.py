from bert_nli import BertNLIModel
from utils.nli_data_reader import NLIDataReader


if __name__ == "__main__":


    model = BertNLIModel(model_path=mpath,batch_size=batch_size,bert_type=bert_type)

    # Read the dataset
    nli_reader = NLIDataReader('./datasets/MQNLI')
    test_data = nli_reader.get_mqnli_examples('0-5gendata-test.json')
