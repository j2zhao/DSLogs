import ast
#import networkx
from astmonkey import visitors, transformers
from logs_decorate import log_function


def ast_playing(source_code):
    node = ast.parse(source_code)
    node = transformers.ParentChildNodeTransformer().visit(node)
    visitor = visitors.GraphNodeVisitor()
    visitor.visit(node)
    for n in ast.walk(node):
        print(type(n))
    visitor.graph.write_png('graph_10.png')
    # print(ast.dump(node))

if __name__ == "__main__":
    with open("./arithmatic_functions.py") as file:
        source_code = file.read()
    ast_playing(source_code)