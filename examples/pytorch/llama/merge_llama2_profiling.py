import os
import json
import argparse
import shutil


def merge_logs(args):
    result = []
    current_min_stamp = float('inf')
    for i in range(args.world_size):
        print(i)
        with open("/workspace/FasterTransformer/build/" + args.profix + '_' + str(i) + '.json') \
                as inputJson:
            current_trace = json.load(inputJson)
            inputJson.close()
            if i == 0:
                for log in current_trace:
                    current_min_stamp = min(log['ts'], current_min_stamp)
            for log in current_trace:
                log['ts'] = log['ts'] - current_min_stamp
            result.extend(current_trace)
    print(len(result))
    with open("/workspace/FasterTransformer/build/model/" + args.profix + '.json', 'w') as outputJson:
        json.dump(result, outputJson)


def main():
    parser = argparse.ArgumentParser(description='Profile-Llama2')
    parser.add_argument('--world-size', type=int, default=4, metavar='N',
                        help='distributed cluster size (default: 4)')
    parser.add_argument('--profix', type=str, default='llama_profile', metavar='S',
                        help='postfix of the tracing file name.')
    args = parser.parse_args()
    merge_logs(args)


if __name__ == '__main__':
    main()