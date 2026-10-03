import time
from flcore.clients.clientflame import clientFLAME
from flcore.servers.serverbase import Server
from threading import Thread

import random
import torch
import numpy as np
import copy
try:
    import hdbscan
except ImportError:
    hdbscan = None
from sklearn.cluster import KMeans, SpectralClustering
class FLAME(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFLAME)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        self.cluster = args.cluster

        # self.load_model()
        self.Budget = []


    def train(self):
        
        for i in range(self.global_rounds+1):
            w_glob = self.global_model.state_dict()
            w_locals = [w_glob for i in range(self.num_clients)]
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
            w_updates = []
        
            cos_list=[]
            local_model_vector = []
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                idx = client.id
                client.train()
                w = client.model.state_dict()
                w_locals[idx] = copy.deepcopy(w)
                w_updates.append(get_update(w, w_glob))
                
            for param in w_locals:
                # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
                local_model_vector.append(parameters_dict_to_vector_flt(param))
            for i in range(len(local_model_vector)):
                cos_i = []
                for j in range(len(local_model_vector)):
                    cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
                    # cos_i.append(round(cos_ij.item(),4))
                    cos_i.append(cos_ij.item())
                cos_list.append(cos_i)
            if(self.cluster == 'kmean'):
                clusterer = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(cos_list)
                label_counts = np.bincount(clusterer.labels_)
                count_label_0 = label_counts[0]
                count_label_1 = label_counts[1]
                benign_label = 1
                if(count_label_0 >= count_label_1):
                    benign_label = 0
                #print("clusterer_labels:",clusterer.labels_)
                benign_client = []
                norm_list = np.array([])

                max_num_in_cluster=0
                max_cluster_index=0
                
                if benign_label == 0:
                    benign_client = np.where(clusterer.labels_ == 0)[0]
                    # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
                elif benign_label == 1:
                    benign_client = np.where(clusterer.labels_ == 1)[0]
                print(benign_client)
            elif(self.cluster == "spectral"):
                clusterer = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0).fit(cos_list)
                label_counts = np.bincount(clusterer.labels_)
                count_label_0 = label_counts[0]
                count_label_1 = label_counts[1]
                benign_label = 1
                if(count_label_0 >= count_label_1):
                    benign_label = 0
                #print("clusterer_labels:",clusterer.labels_)
                benign_client = []
                norm_list = np.array([])

                max_num_in_cluster=0
                max_cluster_index=0
                
                if benign_label == 0:
                    benign_client = np.where(clusterer.labels_ == 0)[0]
                    # norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
                elif benign_label == 1:
                    benign_client = np.where(clusterer.labels_ == 1)[0]
                print(benign_client)
                
            elif(self.cluster == "hdbscan"):
                clusterer = hdbscan.HDBSCAN(min_cluster_size=self.num_join_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
                #print(clusterer.labels_)
                
                benign_client = []
                norm_list = np.array([])

                max_num_in_cluster=0
                max_cluster_index=0
                if clusterer.labels_.max() < 0:
                    for i in range(len(w_locals)):
                        benign_client.append(i)
                        norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(w_updates[i]),p=2).item())
                else:
                    for index_cluster in range(clusterer.labels_.max()+1):
                        if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                            max_cluster_index = index_cluster
                            max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
                    for i in range(len(clusterer.labels_)):
                        if clusterer.labels_[i] == max_cluster_index:
                            benign_client.append(i)
                
                print(benign_client)
        
            for i in range(len(local_model_vector)):
                    # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
                    norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(w_updates[i]),p=2).item())  # no consider BN
            # for i in range(len(benign_client)):
            #     if benign_client[i] < num_malicious_clients:
            #         args.wrong_mal+=1
            #     else:
            #         #  minus per benign in cluster
            #         args.right_ben += 1
            # args.turn+=1
            # print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
            # print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
            
            clip_value = np.median(norm_list)
            for i in range(len(benign_client)):
                gama = clip_value/norm_list[i]
                if gama < 1:
                    for key in w_updates[benign_client[i]]:
                        if key.split('.')[-1] == 'num_batches_tracked':
                            continue
                        w_updates[benign_client[i]][key] *= gama
            w_glob = no_defence_balance([w_updates[i] for i in benign_client], w_glob)
            #add noise
            for key, var in w_glob.items():
                if key.split('.')[-1] == 'num_batches_tracked':
                            continue
                temp = copy.deepcopy(var)
                #temp = temp.normal_(mean=0,std=args.noise*clip_value)
                temp = temp.normal_(mean=0,std=0.001*clip_value)
                var += temp
                
            #self.global_model = copy.deepcopy(w_glob)
            for param_b, param_a in zip(self.global_model.parameters(), w_glob.values()):
                param_b.data.copy_(param_a.data)
                
                
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            # self.receive_models_flame()
            # if self.dlg_eval and i%self.dlg_gap == 0:
            #     self.call_dlg(i)
            # self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFLAME)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
    def receive_models_flame(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    
def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters