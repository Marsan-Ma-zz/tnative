#!/bin/bash

NOW=$(date +"%Y%m%d_%H%M")
PY3='stdbuf -o0 nohup python3 -u'

# [prober]
# $PY3 lib/prober.py _tnative_prober_ > "./logs/nohup_tnative_prober_$NOW.log" &

# [train single model with sample rate]
# $PY3 main.py ffm 0.90 _tnative_ > "./logs/nohup_ffm_$NOW.log" &
# $PY3 main.py xgboost 0.90 _tnative_ > "./logs/nohup_xgboost_$NOW.log" &
# $PY3 main.py sklr 0.90 _tnative_ > "./logs/nohup_sklr_$NOW.log" &
# $PY3 main.py ftrl 0.01 _tnative_ > "./logs/nohup_ftrl_$NOW.log" &

### [feature drop]
# $PY3 main.py feature_drop _tnative_ > "./logs/nohup_feature_drop_$NOW.log" &
# $PY3 main.py drop_exp _tnative_ > "./logs/nohup_drop_exp_$NOW.log" &

### [feature select]
# $PY3 main.py feature_select single _tnative_ > "./logs/nohup_feature_select_single_$NOW.log" &
# $PY3 main.py feature_select interaction _tnative_ > "./logs/nohup_feature_select_interaction_$NOW.log" &

# ## [compare all algs and sample rates]
# $PY3 main.py compare_alg_srate _tnative_ > "./logs/nohup_compare_alg_srate_$NOW.log" &


### [greedy_search parameters]
# $PY3 main.py greedy_search ffm _tnative_ > "./logs/nohup_greedy_search_ffm_D20_$NOW.log" &
# $PY3 main.py greedy_search xgboost _tnative_ > "./logs/nohup_greedy_search_xgboost_$NOW.log" &
# $PY3 main.py greedy_search sklr _tnative_ > "./logs/nohup_greedy_search_sklr_$NOW.log" &
# $PY3 main.py greedy_search skrf _tnative_ > "./logs/nohup_greedy_search_skrf_$NOW.log" &
# $PY3 main.py greedy_search ftrl _tnative_ > "./logs/nohup_greedy_search_ftrl_$NOW.log" &
# $PY3 main.py greedy_search sksgd _tnative_ > "./logs/nohup_greedy_search_sksgd_$NOW.log" &

### [fast greedy_search for model selection]
$PY3 main.py fast_greedy_search ffm _tnative_ > "./logs/nohup_fast_greedy_search_ffm_D20_$NOW.log" &
# $PY3 main.py fast_greedy_search xgboost _tnative_ > "./logs/nohup_fast_greedy_search_xgboost_D20_$NOW.log" &
# $PY3 main.py fast_greedy_search sklr _tnative_ > "./logs/nohup_fast_greedy_search_sklr_D20_$NOW.log" &
# $PY3 main.py fast_greedy_search skrf _tnative_ > "./logs/nohup_fast_greedy_search_skrf_D20_$NOW.log" &
# $PY3 main.py feature_rfecv _tnative_ > "./logs/nohup_rfecv_D20_$NOW.log" &

### [remove_blacklist]
# $PY3 main.py _tnative_remove_blacklist_ > "./logs/nohup_remove_blacklist_$NOW.log" &

### [bagging]
# $PY3 main.py bagging_layer1 sklr _tnative_bagging_layer1_ > "./logs/nohup_tnative_bagging_layer1_sklr_$NOW.log" &
# $PY3 main.py bagging_layer1 xgboost _tnative_bagging_layer1_ > "./logs/nohup_tnative_bagging_layer1_xgboost_$NOW.log" &
# $PY3 lib/booster.py layer_1_sklr _tnative_booster_collect_ > "./logs/nohup_tnative_booster_collect_$NOW.log" &
# $PY3 main.py booster_layer2 sklr _tnative_ > "./logs/nohup_fast_greedy_search_sklr_D20_$NOW.log" &


### [ensemble]
# $PY3 lib/ensembler.py 5_fold_xgboost 0.1 _tnative_ > "./logs/nohup_5_fold_xgboost_$NOW.log" &
# $PY3 lib/ensembler.py 5_fold_sklr 0.1 _tnative_ > "./logs/nohup_5_fold_sklr_$NOW.log" &
# $PY3 lib/ensembler.py xgboost_sklr 0.1 _tnative_ > "./logs/nohup_xgboost_sklr_$NOW.log" &
# $PY3 lib/ensembler.py dnq_ensemble 0.1 status _tnative_ > "./logs/nohup_dnq_status_$NOW.log" &
# $PY3 lib/ensembler.py dnq_ensemble 0.1 lang _tnative_ > "./logs/nohup_dnq_lang_$NOW.log" &
# $PY3 lib/ensembler.py dnq_ensemble 0.1 domain _tnative_ > "./logs/nohup_dnq_domain_$NOW.log" &
# $PY3 lib/ensembler.py dnq_ensemble 0.1 all _tnative_ > "./logs/nohup_dnq_domain_$NOW.log" &

### [generate submit file]
# $PY3 lib/submit.py submit xgboost_t90_v9_auc_886_20150909_0845 > "./logs/nohup_submit_xgboost_t90_v9_auc_886_20150909_0845_$NOW.log" &
# $PY3 lib/submit.py submit sklr_t90_v9_auc_972_20151002_1359 > "./logs/nohup_sklr_t90_v9_auc_972_20151002_1359_$NOW.log" &
# $PY3 lib/submit.py submit ffm_t90_v9_auc_923_20150912_2203 > "./logs/nohup_ffm_t90_v9_auc_923_20150912_2203_$NOW.log" &
# $PY3 lib/submit.py fast_data_submit sklr_t90_v9_auc_948_20151011_2336 D_20_all_submit > "./logs/nohup_sklr_t90_v9_auc_948_20151011_2336_$NOW.log" &
# $PY3 lib/submit.py fast_data_submit ffm_t90_v9_auc_947_20151012_0046 D_20_all_submit > "./logs/nohup_ffm_t90_v9_auc_947_20151012_0046_$NOW.log" &
# $PY3 lib/submit.py fast_data_layer1 ffm_t90_v9_auc_947_20151012_0046 D_20_all > "./logs/nohup_ffm_t90_v9_auc_947_20151012_0046_train_$NOW.log" &

### [train & submit]
# $PY3 lib/submit.py train_n_submit _train_n_submit_ > "./logs/nohup_train_n_submit_$NOW.log" &

