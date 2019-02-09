#include "lab1_sequential.h"
#include "stdio.h"
#include "time.h"
#include "limits.h"
#include "malloc.h"
#include "stdlib.h"
#include "math.h"

void updatecentroid(int N,int K,int* data_points, int** data_point_cluster, float** centroids, int iteration );
void random_restart(int N, int K, int* data_points, float**centroids);
int nearest_centroid(int i, int K, int* data_points, float**centroids, int iteration_done);
int cal_distance(int x, int y, int z, int x1, int y1, int z1);


void kmeans_sequential(int N, int K, int* data_points, int** data_point_cluster, float** centroids,int* num_iterations){
    *centroids = (float*)malloc(sizeof(int)*(K*3*101));
    *data_point_cluster = (int*)malloc(sizeof(int)*(N*4));
    *(num_iterations) =1;
    random_restart(N,K,data_points,centroids);

    int j=0;
    for(int i=0;i<3*N;i=i+3){
        *(*data_point_cluster+j) = *(data_points+i);
        *(*data_point_cluster+j+1) = *(data_points+i+1);
        *(*data_point_cluster+j+2) = *(data_points+i+2);
        *(*data_point_cluster+j+3) = 0;
        j=j+4;
    }
    
    while(*(num_iterations)<=100){
        //calculate distance of each point with all centroids and assign them to nearest centroid and update in data_point_cluster.
        
        j=0;
        for(int i=0;i<3*N;i=i+3){
            int k = nearest_centroid(i,K,data_points,centroids,*(num_iterations)-1);
            *(*data_point_cluster+j+3) = k;
            j=j+4;        
        }
        //calculate mean of all points of same datapoints and recompute centroids

        updatecentroid(N, K, data_points,data_point_cluster,centroids,*(num_iterations));
        *(num_iterations) +=1;        
    } 
    *(num_iterations)= *(num_iterations)-1;
}
void updatecentroid(int N,int K,int* data_points, int** data_point_cluster, float** centroids, int iteration ){

    int array[K];
    for(int i=0;i<4*N;i=i+4){
        int id = *(*data_point_cluster+i+3);
        int x = *(*data_point_cluster+i);
        int y = *(*data_point_cluster+i+1);
        int z = *(*data_point_cluster+i+2);
        *(*centroids+3*id+K*3*iteration) += x;
        *(*centroids+3*id+1+K*3*iteration) += y;
        *(*centroids+3*id+2+K*3*iteration) += z;
        array[id] +=1;
    }
    for(int i=0;i<K*3;i=i+3){
        *(*centroids+K*3*iteration+i) = (float) *(*centroids+K*3*iteration+i)/array[i/3];
        *(*centroids+K*3*iteration+i+1) = (float) *(*centroids+K*3*iteration+i+1)/array[i/3];
        *(*centroids+K*3*iteration+i+2) = (float) *(*centroids+K*3*iteration+i+2)/array[i/3];
    }
}
void random_restart(int N, int K, int* data_points, float**centroids){
    srand(time(0));
    for(int i=0;i<K*3;i= i+3){
        int randomno = rand() % (N-1);
        *(*centroids+i) = *(data_points+randomno);
        *(*centroids+i+1) = *(data_points+1+randomno);
        *(*centroids+i+2) = *(data_points+2+randomno);
    }
}
int nearest_centroid(int i, int K, int* data_points, float**centroids, int iteration_done){
    int x = (int)*(data_points+i);
    int y = (int)*(data_points+i+1);
    int z = (int)*(data_points+i+2);
    int id;
    int distance = INT_MAX;
    for(int j=0;j<K*3;j=j+3){
        int x1 = (int)*(*centroids+j+K*3*iteration_done);//update it for another iteration
        int y1 = (int)*(*centroids+j+1+K*3*iteration_done);
        int z1 = (int)*(*centroids+j+2+K*3*iteration_done);
        if(cal_distance(x,y,z,x1,y1,z1)<distance){
            id = j/3;
            distance = cal_distance(x,y,z,x1,y1,z1); 
        }
    }
    return id;
}
int cal_distance(int x, int y, int z, int x1, int y1, int z1){
    int s =0;
    int x2 = (x-x1)*(x-x1);
    int y2= (y-y1)*(y-y1);
    int z2= (z-z1)*(z-z1);
    return (int)sqrt(x2+y2+z2);
}
