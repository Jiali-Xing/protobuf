# test_globals.py
def get_slo(method, tight):
    return 200 if tight else 100

def main():
    global method, SLO, tightSLO, skipOptimize
    print(f"method: {method}, SLO: {SLO}, tightSLO: {tightSLO}, skipOptimize: {skipOptimize}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('method', type=str, help='Specify the method')
    parser.add_argument('--tight-slo', action='store_true', default=False, help='Enable tight SLO')
    parser.add_argument('--skip-opt', action='store_true', default=False, help='Skip optimization')
    args = parser.parse_args()

    global method, SLO, tightSLO, skipOptimize

    method = args.method
    method_map = {
        's1': 'S_102000854',
        's2': 'S_149998854',
        's3': 'S_161142529',
    }
    method = method_map.get(method, method)
    tightSLO = args.tight_slo
    skipOptimize = args.skip_opt

    SLO = get_slo(method, tight=tightSLO)

    main()
