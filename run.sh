python main.py > output/no_topk.log
python main.py --topk_approx 32 > output/topk32.log
python main.py --topk_approx 64 > output/topk64.log
python main.py --topk_approx 128 > output/topk128.log