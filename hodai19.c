#include "nn.h"
#include <time.h>

//補題１
//x:m*n行列を出力
void print(int m, int n, const float * x){
    for (int i = 0; i < m*n; i++){
        printf("%f ", x[i]);
        if((i+1)%n==0){
            printf("\n");
        }
    }
}

//補題２
//配列oにA*xを代入A:m*n行列,x:n*1行列
//o=Ax
void mul(int m, int n, const float *x, const float *A, float *o){
    for (int i = 0; i < m;i++){
        o[i] = 0;
        for (int j = 0; j < n;j++){
            o[i] += A[i * n + j] * x[j];
        }
    }
}

//FC層
//A[m*n],b[m],o=A*x+b
void fc(int m, int n, const float * x, const float *A, const float * b, float *o){
    mul(m, n, x, A, o);
    for (int i = 0; i < m; i++){
        float a = o[i] + b[i];
        o[i] = a;
    }
}

//補題3
//Relu関数(活性化関数)
//x[n],y[n]
//xを入力,yを出力
//y={x(x>=0),0(x<0)}
void relu(int n, const float *x, float *y){
    for (int i = 0; i < n; i++){
        if(x[i]>=0){
            y[i] = x[i];
        }
        else{
            y[i] = 0;
        }
    }
}

//補題4
//softmax関数(出力層)
//x[n],y[n]
//xを入力,yを出力
//y=exp(x-x_max)/sigma{exp(x-x_max)}
void softmax(int n, const float *x, float *y){
    float sum=0;
    float x_max = 0;
    for (int i = 0; i < n; i++){
        if(x_max < x[i]){
            x_max = x[i];
        }
    }
        for (int i = 0; i < n; i++)
        {
            float a = exp(x[i]-x_max);//オーバーフロー防止
            sum += a;
        }
    for (int i = 0; i < n; i++){
        float a = exp(x[i]-x_max) / sum;//オーバーフロー防止
        y[i] = a;
    }
}


//補題8
//softmax層の誤差逆伝搬法
//dEdx=y-t
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx){
    for (int i = 0; i < n; i++){
        if(i == t){
            dEdx[i] = y[i] - 1;
        }
        else{
            dEdx[i] = y[i];
        }
    }
}

//補題9
//Relu層の誤差逆伝搬法
//dEdx={dEdy(dEdy>0),0(dEdy<=0)}
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx){
    for (int i = 0; i < n; i++){
        if(x[i] >0){
            dEdx[i] = dEdy[i];
        }
        else{
            dEdx[i] = 0;
        }
    }
}

//補題10
//A=m*n行列とする
//FC層の誤差逆伝搬法
//dEdA=dEdy*x,dEdb=dEdy,dEdx=A*dEdy
void fc_bwd(int m, int n, const float *x, const float *dEdy,const float *A,float *dEdA,float*dEdb,float*dEdx){
    for (int i = 0; i < m; i++){
        dEdb[i] = dEdy[i];
        for (int j = i*n; j < (i+1)*n; j++){
            float a = dEdy[i] * x[j - i * n];
            dEdA[j] = a;
        }
    }
   

    for (int i = 0; i < n; i++){
        dEdx[i] = 0;
        for (int j = 0; j < m ; j++){
            float a = A[j * n + i] * dEdy[j];
            dEdx[i] += a; 
        }
    }
}


//補題12
//a,bの値を入れ替える
void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

//ランダムシャッフル
//x[n]
//xの要素をシャッフル
void shuffle(int n, int *x){
    srand(time(NULL));
    for (int i = 0; i < n;i++){
        int num = rand()*rand() % n;//RAND_MAX=32767より60000に足りないからrand()*rand()にする
        swap(&x[i], &x[num]);
    }
}

//補題13
//交差エントロピー誤差
//-log(y+1e-7)を計算する
float cross_entropy_error(const float*y,int t){
    return -log(y[t]+1e-7);//1e-7を加えることでオーバーフローを防ぐ
}

//補題14
//配列の加算
//o+=x
//o[n],x[n]
void add(int n, const float *x, float*o){
    for (int i = 0;i < n;i++){
        float a = o[i] + x[i];
        o[i] = a;
    }
}

//配列の引き算
//o-=x
//o[n],x[n]
void mainasu(int n,const float *x,float*o){
    for (int i = 0;i < n;i++){
        float a = o[i] - x[i];
        o[i] = a;
    }
}

//配列oの各要素ををx倍にする
//o[n]
void scale(int n,float x,float*o){
    for (int i = 0; i < n;i++){
        float a = o[i] * x;
        o[i] = a;
    }
}

//配列の初期化
//o[n]
//oをxで初期化
void init(int n,float x,float*o){
    for (int i = 0;i < n;i++){
        o[i] = x;
    }
}

//一様分布による初期化
//o[n]
//oの各要素を[-1,1]で初期化
void rand_init(int n, float* o){
    srand(time(NULL));
    for (int i = 0; i < n; i++){
        float x = (double)rand()/RAND_MAX;
        int y = rand();
        if(y%2 == 0){
            x = -x;
        }
        o[i] = x;
    }
}

//0<x<1の乱数生成
double Uniform( void ){
    return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
}

//Box-muller法 正規分布乱数 mu:平均値 sigma:標準偏差
double rand_normal( double mu, double sigma ){
    double z=sqrt( -2.0*log(Uniform()) ) * sin( 2.0*M_PI*Uniform() );
    return mu + sigma*z;
}

//Heの初期値による初期化
//o[n]
void he_init(int n, float*o){
    srand(time(NULL));
    for (int i = 0; i < n; i++){
        o[i] = rand_normal(0, sqrt(2.0/n));//平均0,標準偏差sqrt(2.0/n)の正規分布
    }
}

//Xavierの初期値による初期化
//o[n]
void xavier_init(int n, float*o){
    srand(time(NULL));
    for (int i = 0; i < n; i++){
        o[i] = rand_normal(0, sqrt(1.0/n));//平均0,標準偏差sqrt(1.0/n)の正規分布
    }
}

//標準偏差0.01の正規分布乱数による初期化
//o[n]
void normal_init(int n, float *o){
    srand(time(NULL));
    for (int i = 0; i < n; i++){
        o[i] = rand_normal(0, 0.01);//平均0,標準偏差0.01の正規分布
    }
}

//6層NNによる推論
//A:重み,b:バイアス,x:入力,y:出力
int inference6(const float*A1,const float *b1,const float*A2,const float*b2,const float*A3,const float*b3, const float *x,float *y){
    float *y1 = malloc(sizeof(float) * 50);
    float *y2 = malloc(sizeof(float) * 100);
    
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    fc(100, 50, y1, A2, b2, y2);
    relu(100, y2, y2);
    fc(10, 100, y2, A3, b3, y);
    softmax(10, y, y);
    int m = 0;
    float max = 0;
    for (int i = 0; i < 10; i++){
        if(max < y[i]){
            m = i;
            max = y[i];
        }
    }
    free(y1);
    free(y2);
    return m;
}

//6層NNによる誤差逆伝搬法
//relux:順伝搬のrelu層の入力,dE~:偏微分,A:重み,b:バイアス,x:入力,t:正解,y:relu関数,softmaxの出力
void backward6(float *relu1x,float *relu2x,float *dEdx10,float *dEdx100,float *dEdx50,float *dEdx784,const float *A1, const float *b1,const float *A2,const float *b2, const float *A3, const float *b3, const float *x, unsigned char t, float *y1,float*y2,float*y3, float *dEdA1, float *dEdb1,float *dEdA2,float*dEdb2,float*dEdA3,float*dEdb3){

    
    fc(50, 784, x, A1, b1, relu1x);
    relu(50, relu1x, y1);

    
    fc(100, 50, y1, A2, b2, relu2x);
    
    relu(100, relu2x, y2);
    fc(10, 100, y2, A3, b3, y3);
    softmax(10, y3, y3);

    
    softmaxwithloss_bwd(10, y3, t, dEdx10);
    
    fc_bwd(10,100 , y2, dEdx10, A3, dEdA3, dEdb3, dEdx100);

    relu_bwd(100, relu2x, dEdx100, dEdx100);
    
    fc_bwd(100, 50 , y1, dEdx100, A2, dEdA2, dEdb2, dEdx50);

    relu_bwd(50, relu1x, dEdx50, dEdx50);
    
    fc_bwd(50,784 , x, dEdx50, A1, dEdA1, dEdb1, dEdx784);

}

//ファイルにセーブ
//filename:書き込むファイル,A[m*n],b[n]
void save(const char *filename, int m, int n, const float *A, const float*b){
    FILE *fp;
    if((fp = fopen(filename,"wb"))==NULL){
        printf("\aファイルをオープンできません。\n");
    }
    else{
        fwrite(A, sizeof(float), m * n, fp);
        fwrite(b, sizeof(float), n, fp);
        fclose(fp);
    }
}

//adagradでパラメータ更新
void adagrad(float n,float x,float *h, float *da,float *a){//x:学習係数,n:size,da:average
    //平均化
    scale(n, 0.01, da);
    //h更新
    for (int i = 0; i < n; i++){
        h[i] += da[i] * da[i];
        a[i] -= x * da[i] / sqrt(h[i] + 1e-7);//発散防止
    }
}

//SGDでパラメータ更新
void sgd(float n, float x,float *da, float *a ){//n:size,x:learning rate
    scale(n, 0.01, da);
    scale(n, x, da);
    mainasu(n, da, a);
}

//Momentumでパラメータ更新
void momentum(float n, float x, float k, float *da, float *a,float *v){//k=0.9,n:size,x:learning rate
    scale(n, 0.01, da);
    scale(n, k, v);
    scale(n, x, da);
    mainasu(n, da, v);
    add(n, v, a);
}

//filename:ファイルの名前
//ファイルをロード
void load(const char *filename, int m, int n, float *A, float *b){
    FILE *fp;
    if((fp=fopen(filename,"rb"))==NULL){
        printf("\aファイルをオープンできません。\n");
    }
    else{
        fread(A, sizeof(float), m * n, fp);
        fread(b, sizeof(float), n, fp);
    }
    fclose(fp);
}

//推論の結果を棒グラフにする
void graph(int n, const float *x){
    printf("\nnum|probability|graph\n");
    for (int i = 0; i < n; i++){
        printf("%3d|%10.2f%%|", i, x[i] * 100);
        for (int j = 1; j <= x[i] * 100;j++){
            printf("=");
        }
        printf("\n");
    }
}

int main(int argc, char const *argv[]){
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;

    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;

    int width = -1;
    int height = -1;

    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);
    int i = atoi(argv[5]);
    save_mnist_bmp(train_x + 784 * i, "train_%05d.bmp", i);
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);
    float *x = load_mnist_bmp(argv[4]);
    float *yre = malloc(sizeof(float) * 10);
    if(argc != 6){
        printf("error");
        exit(1);
    }
    load(argv[1], 784, 50, A1, b1);
    load(argv[2], 50, 100, A2, b2);
    load(argv[3], 100, 10, A3, b3);
    int m = inference6(A1, b1, A2, b2, A3, b3, x, yre);
    graph(10, yre);
    printf("the answer is %d", m);
    return 0;
}