# CHMSL-blend
"An effective coarse-to-fine color correction algorithm for multi-view images" (2023) by Kuo-Liang Chung, Ting-Chung Tang and Yu-Che Liu.

<div align=center>
<img src="https://github.com/ivpml84079/CHMSL-blend/blob/main/Fig/example.png">
</div>

The visual quality improvement of our algorithm. (a) The stitched three multi-view images after performing the seam cutting method by [Yuan et al.](https://ieeexplore.ieee.org/abstract/document/9115682) on the selected testing image set. on the selected testing image set. (b) The two magnified sub-images of (a). (c) The [HM](https://ieeexplore.ieee.org/document/4539698) method. (d) The [AGL](https://www.sciencedirect.com/science/article/pii/S0924271617300990?via%3Dihub) method. (e) The [GP](https://www.sciencedirect.com/science/article/pii/S0924271619302151?via%3Dihub) method. (f) The [HHM](https://ieeexplore.ieee.org/document/9261383) method. (g) The [GJBI](https://ieeexplore.ieee.org/document/8676030) method. (h) The [LJBI](https://www.mdpi.com/2072-4292/14/21/5440) method. (i) The [HoLoCo](https://www.sciencedirect.com/science/article/pii/S1566253523000672?via%3Dihub) method. (j) Ours for the HM method. (k) Ours for the AGL method. (l) Ours for the GP method. (m) Ours for the HHM method. (n) Ours for the GJBI method. (o) Ours for the LJBI method. (p) Ours for the HoLoCo method.

## usage

#### How to Adjust Program Parameters

In the main program "main.cpp", parameters can be set in lines 17-30. 

## Enviroment
* Windos 10 64-bit
* Visual Studio 2019
* Visual Studio platform toolset LLVM-clang
* ISO C++ 17 

## Other high resolution images of blended results
https://drive.google.com/drive/folders/1zX1IsOWIvnIBhxuKZpFM5ZOOM7yVLJKo?usp=sharing

## Appendix
[The HoLoCo method's retrain model](https://drive.google.com/drive/folders/1aILWJX0GkDqo2Qu6ffRkSg-jG2Oyxl_P?usp=sharing)
