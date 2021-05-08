import ast
#import networkx
from astmonkey import visitors, transformers
from numpy.lib.arraysetops import isin

# TODO: Support case where the log is not just one name
def gen_lognode(func_name, attr = None):

    target = ast.Name(id = func_name, ctx = ast.Store())
    value = ast.Call(func = ast.Name(id = 'log_function', ctx = ast.Load()), args= [ast.Name(id = func_name, ctx = ast.Load())], keywords=[])
    node = ast.Assign(targets = [target], value = value, type_comment = None)
    return node

def get_name(name_node):
    '''
    Returns:

    name: name of the function in a string
    path: if the function is not a "first-order" variable, return the full AST path
    skip: eventually support more complicated skips?
    '''
    if isinstance(name_node, ast.Name):
        if name_node.id == 'LoggedNDArray':
            return None, None, True
        return name_node.id, None, False
        
    else:
        return None, None, True


def ast_annotate(source_code):
    node = ast.parse(source_code)
    temp = []
    for i, n in enumerate(node.body):
        for m in ast.walk(n):
            if isinstance(m, ast.Call):
                func_name, att, skip = get_name(m.func)
                if skip:
                    continue
                new_n = gen_lognode(func_name, att)
                print(ast.dump(new_n))
                temp.append((i, new_n))
    offset = 0
    for i, n in temp:
        node.body.insert(i + offset, n)
        offset += 1
    
    node = transformers.ParentChildNodeTransformer().visit(node)
    ast.fix_missing_locations(node)
    exec(compile(node, filename="<ast>", mode="exec"))
    visitor = visitors.GraphNodeVisitor()
    visitor.visit(node)
    visitor.graph.write_png('graph_4.png')
    print('hello')
    generated_code = visitors.to_source(node)
    print(generated_code)
    with open('new_code.py', 'w') as f:
        f.write(generated_code)    
    #print(ast.dump(node))

if __name__ == "__main__":
    with open("./decorated_example.py") as file:
        source_code = file.read()
    ast_annotate(source_code)