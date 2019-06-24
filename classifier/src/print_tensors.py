import tensorflow as tf
import argparse

def print_tensors(pb_file):
  """Iterate through graph nodes and print node names.

  Optionally, the return value of the function (a list of node names)
  may be captured in a variable and used for other purposes.
  """
  with tf.gfile.GFile(pb_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

  for op in graph.get_operations():
    print(op.name)

  return [op.name for op in graph.get_operations()]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dir',
      type=str,
      default='',
      help='Path to the directory that holds your model'
  )
  parser.add_argument(
      '--name',
      type=str,
      default='nodes.txt',
      help='Name of the output file'
  )
  args, unparsed = parser.parse_known_args()
  names = print_tensors(args.dir)
  # print(names)
  with open(args.name, "w+") as file:
     for name in names:
         file.write(name)
         file.write('\n')
     file.close()
