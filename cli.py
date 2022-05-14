import argparse
import open3d

import registration


def main(args):
    source = open3d.io.read_triangle_mesh(args.source)
    target = open3d.io.read_triangle_mesh(args.target)
    source.compute_vertex_normals()
    target.compute_vertex_normals()

    if args.method == "icp":
        deformed = registration.icp(source, target)
    elif args.method == "nricp":
        deformed = registration.nricp(source, target, coverage=True)
    else:
        deformed = source

    if args.destination != "":
        open3d.io.write_triangle_mesh(
            args.destination, deformed, write_vertex_normals=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", dest="method", type=str, default="icp")
    parser.add_argument("--src", dest="source", type=str, default="data/source.obj")
    parser.add_argument("--tgt", dest="target", type=str, default="data/target.obj")
    parser.add_argument("--dest", dest="destination", type=str, default="")
    args = parser.parse_args()

    main(args)
