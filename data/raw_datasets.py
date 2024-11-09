from datasets import load_dataset, load_from_disk
import os


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        # default load from disk
        if "Anthropic/hh-rlhf" in dataset_name:
            self.raw_datasets = load_from_disk(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_answer(self, sample):
        return

    def get_prompt_and_answer(self, sample):
        return


class LocalJsonFileDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, for_backbone=False):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local_jsonfile"
        self.dataset_name_clean = "jsonfile"
        assert os.path.exists(dataset_name), f"Not found, plz check path {dataset_name}!"
        self.for_backbone = for_backbone
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             dataset_name + '/train.json',
                                             "eval":
                                             dataset_name + '/eval.json',
                                             "test":
                                             dataset_name + '/test.json',
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    def get_test_data(self):
        if self.raw_datasets['test'] is not None:
            return self.raw_datasets['test']
        return None

    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return sample['prompt']
        return None

    def get_answer(self, sample):
        if sample['answer'] is not None:
            return sample['answer']
        return ''

    def get_prompt_and_answer(self, sample):
        if sample['prompt'] is not None and sample['answer'] is not None:
            return sample['prompt'] + "\n" + sample['answer']
        return None

