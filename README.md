# dmt

Computes the [max-tree](https://scikit-image.org/docs/dev/auto_examples/developers/plot_max_tree.html) of an image - a hierarchical representation of that image that is the basis for a large family of morphological filters. It is often used as the basis of feature engineering in remote sensing applications.

dmt is written in C++ and may be used as shared library and command line tool.  

## Dependencies

dmt requires the following dependencies. Please make sure, that these are installed, before attempting to compile the code.

* CMake 3.10+
* C++11 compliant compiler (e.g. g++ 4.9+)
* OpenMP 4.0+ (e.g. g++ 4.9+)
* HDF5 1.8+
* Message Passing Interface (MPI) 2.0+

## Compilation

dmt follows the standard CMake project conventions. Create a build directory, change to it, generate the build script and compile it. A convenience short-hand can be found below.

``` bash
mkdir build && cd build && cmake .. && make
```

The provided CMake script checks, but does not install, all of the necessary dependencies listed above.

## Usage

dmt's command line usage flags are shown below. You may obtain the same message by invoking `dmt -h`:

```
dmt - distributed max-tree
Usage:
```

The typical basic usage of dmt is shown below. The line shows a typical high-performance computing setup with multiple distributed nodes and processing cores per node, but also runs on your work stations or laptop. The data is passed to the application in form of an HDF5 file. 

``` bash
mpirun -np <NODES> ./dmt -t <THREADS> <PATH_TO_HDF5_FILE>
```

## Citation

If you wish to cite dmt in your academic work, please use the following reference:

Plain reference
```
Götz, M., Cavallaro, G., Géraud, T., Book, M., & Riedel, M. (2018). 
Parallel computation of component trees on distributed memory machines. 
IEEE Transactions on Parallel and Distributed Systems, 29(11), 2582-2598.
```

BibTex
``` bibtex
@article{gotz2018parallel,
  title={Parallel computation of component trees on distributed memory machines},
  author={G{\"o}tz, Markus and Cavallaro, Gabriele and G{\'e}raud, Thierry and Book, Matthias and Riedel, Morris},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  volume={29},
  number={11},
  pages={2582--2598},
  year={2018},
  publisher={IEEE}
}
```

## Contact

If you want to let us know about feature requests, bugs or issues you are kindly referred to the [issue tracker](https://github.com/Markus-Goetz/dmt/issues).

For any other discussion, please write an [e-mail](mailto:markus.goetz@kit.edu).

