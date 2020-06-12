import json
from basepairmodels.common import training

class ProfileModel:
    
    def __init__(self, input_seq_len=1346, output_len=100, num_tasks=2):
        self.input_seq_len = input_seq_len
        self.output_len = output_len
        self.num_tasks = num_tasks
        
        
   
    def _set_input_params(self, input_params):
        self.input_params = input_params
        
    def _set_output_params(self, output_params):
        self.output_params = output_params

    def _set_genome_params(self, genome_params):
        self.genome_params = genome_params
    
    def _set_batch_gen_params(self, batch_gen_params):
        self.batch_gen_params = batch_gen_params
    
    def _set_hyper_params(self, hyper_params):
        self.hyper_params = hyper_params
        
    def _set_parallelization_params(self, parallelization_params):
        self.parallelization_params = parallelization_params
    
    def _set_network_params(self, network_params):
        self.network_params = network_params
        
    def set_params(self, params_json):
        with open(params_json, 'r') as params:
            self.params = json.loads(params.read())
        self._set_input_params(self.params['input_params'])
        self._set_output_params(self.params['output_params'])
        self._set_genome_params(self.params['genome_params'])
        self._set_batch_gen_params(self.params['batch_gen_params'])
        self._set_hyper_params(self.params['hyper_params'])
        self._set_parallelization_params(self.params['parallelization_params'])
        self._set_network_params(self.params['network_params'])

        
        
    def train_and_validate(self, train_chroms, val_chroms, 
                           model_dir='.', suffix_tag=None):
        
        
        self.model = training.train_and_validate(
            self.input_params, self.output_params, self.genome_params,
            self.batch_gen_params, self.hyper_params, 
            self.parallelization_params, self.network_params, train_chroms, 
            val_chroms, model_dir=model_dir, suffix_tag=suffix_tag)
        

        
        
    def predict():
        """
            predict with batch generators using json input data
        """
        
        
    def predict(sequences, batch_size):
        """
            predict on list of input sequences
            
            add control for bpnet
        """
        
        
    def compute_metrics(labels_bw, predictions_bw, locations, metrics):
        """
            compute specified metrics between labels and predictions 
            bigwigs at given locations 
        """
    
    def compute_metrics(labels, predictions, metrics):
        """
            compute specified metrics between lists of labels and 
            predictions
        """
    
    def compute_importance_scores(locations, numshuffles):
        """
        
                     add control for bpnet
        """
        
    def compute_importance_scores(sequences, numshuffles):
        """
        
        
                    add control for bpnet
        """
        
       
        
        
    