# rparis6k, roxford5k
python3 evaler/example_evaluate.py roxford5k R50_CFCD_D512/checkpoints/model_s2_0050_roxford5k.pickle
python3 evaler/example_evaluate.py rparis6k  R50_CFCD_D512/checkpoints/model_s2_0050_rparis6k.pickle

# +1M results
python3 evaler/example_evaluate.py roxford5k R50_CFCD_D512/checkpoints/model_s2_0050_roxford5k.pickle R50_CFCD_D512/checkpoints/model_s2_0050_revisitop1m.mat 1
python3 evaler/example_evaluate.py rparis6k  R50_CFCD_D512/checkpoints/model_s2_0050_rparis6k.pickle R50_CFCD_D512/checkpoints/model_s2_0050_revisitop1m.mat 1

