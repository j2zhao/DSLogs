"""
"""
import ast
import networkx as nx
import sys
import matplotlib.pyplot as plt
import pickle
#{'time': '1615516610.6003299', 'context': '<module>,307', 'function_name': 'add,__call__', 'input_ids': '((100, 10), 140623922695184),1', 'output_ids': '((100, 10), 140623924935360)', 'args': 'None'}
# 1615516610.600235;new;((100, 10), 140623924935136)
#{'time': '1615529508.270748', 'filename': 'test_logged_array.py,24', 'context': ['test = (test + 1) * 2\n'], 'function_name': 'multiply,__call__', 'input_ids': '((100, 10), 4973174992),2', 'output_ids': '((100, 10), 4973178320)', 'args': 'None'}
def parse_logs(file_name, graph_name):
    with open(file_name, 'r') as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    
    logs = []
    for line in content:
        log_ = line.split(';')
        if line[0] == '{':
            log = ast.literal_eval(line) 
            logs.append(log)
        elif log_[1] == 'new':
            log = {}
            log['time'] = log_[0]
            log['function_name'] = log_[1]
            log['input_ids'] = log_[2]
            logs.append(log)
        else:
            log = {}
            log['time'] = log_[0]
            log['function_name'] = log_[1]
            log['input_ids'] = log_[2]
            log['output_ids'] = log_[3]
            logs.append(log)
    # sort by time
    logs.sort(key= lambda x : float(x['time']))
    graph = nx.DiGraph()
    for log in logs:
        if log['function_name'] == 'new':
            arr = ast.literal_eval(log['input_ids'])
            if str(arr) not in graph:
                graph.add_node(str(arr), shape= arr[0], function_name = 'new')
        elif log['function_name'] == 'relation':
            arr = ast.literal_eval(log['output_ids'])
            if str(arr) not in graph:
                graph.add_node(str(arr), shape= arr[0], function_name = 'relation')
            input = ast.literal_eval(log['input_ids'])
            if str(input) not in graph:
                graph.add_node(str(input), shape= input[0], function_name = 'new')
            graph.add_edge(str(input), str(arr))

        else:
            # if log['function_name'] == '__getitem__':
            #     continue
            name = log['context'].split(',')[-1] + ',' + log['function_name']
            if isinstance(log['input_ids'], list):
                inputs = [str(x) for x in log['input_ids']]
            else:
                inputs = [str(log['input_ids'])]
            if isinstance(log['output_ids'], list):
                outputs = [str(x) for x in log['output_ids']]
            else:
                outputs = [str(log['output_ids'])]
            outputs = log['output_ids']
            graph.add_node(name, function_name = log['function_name'], nline = log['context'].split(',')[-1], inputs = inputs, outputs = outputs)
            input_ids = log['input_ids']
            if not isinstance(input_ids,list):
                id = input_ids[1]
                shape = input_ids[0]
                if str(input_ids) not in graph:
                    graph.add_node(str(input_ids), id = id, shape=shape, function_name = 'new')
                graph.add_edge(str(input_ids), name)
            else:
                for input_id in input_ids:
                    if not isinstance(input_id, tuple):
                        continue
                    id = input_id[1]
                    shape = input_id[0]
                    if str(input_id) not in graph:
                        graph.add_node(str(input_id), id = id, shape=shape, function_name = 'new')
                    graph.add_edge(str(input_id), name)
            
            output_ids = log['output_ids']
            if not isinstance(output_ids,list):
                id = output_ids[1]
                shape = output_ids[0]
                if str(output_ids) not in graph:
                    graph.add_node(str(output_ids), id = id, shape=shape, function_name = 'new')
                graph.add_edge(name, str(output_ids))
            else:
                for out in output_ids:
                    if not isinstance(out, tuple):
                        continue
                    id = out[1]
                    shape = out[0]
                    if str(out) not in graph:
                        graph.add_node(str(out), id = id, shape=shape, function_name = 'new')
                    graph.add_edge(name, str(out))

    nx.draw(graph, pos = nx.planar_layout(graph), with_labels=True)
    plt.show()
    with open(graph_name, 'wb') as f:
        pickle.dump(graph, f)

            
                
if __name__ == '__main__':
    file_name = sys.argv[1]
    graph_name = sys.argv[2]
    parse_logs(file_name, graph_name)
