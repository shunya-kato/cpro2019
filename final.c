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
//dEdx={dEdy(x>0),0(x<=0)}
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
//0[n]
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


int main(int argc, char const *argv[]){
    if(argc != 4){
        printf("error");
        exit(1);
    }

    
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

    // これ以降，３層NN の係数 A_784x10 および b_784x10 と，
    // 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
    // テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
    // を使用することができる．

    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);
    int judge;
    printf("your judge is 1 => rand_init\n");
    printf("your judge is 2 => he_init\n");
    printf("your judge is 3 => normal_init\n");
    printf("your jugde is 4 => xavier_init\n");
    printf("your judge is ");
    scanf("%d", &judge);
    printf("\n");
    if(judge == 1){
        rand_init(39200, A1);
        rand_init(50, b1);
        rand_init(5000, A2);
        rand_init(100, b2);
        rand_init(1000, A3);
        rand_init(10, b3);
    }
    else if(judge == 2){
        he_init(39200, A1);
        he_init(50, b1);
        he_init(5000, A2);
        he_init(100, b2);
        he_init(1000, A3);
        he_init(10, b3);
    }
    else if(judge == 3){
        normal_init(39200, A1);
        normal_init(50, b1);
        normal_init(5000, A2);
        normal_init(100, b2);
        normal_init(1000, A3);
        normal_init(10, b3);
    }
    else if(judge == 4){
        xavier_init(39200, A1);
        xavier_init(50, b1);
        xavier_init(5000, A2);
        xavier_init(100, b2);
        xavier_init(1000, A3);
        xavier_init(10, b3);
    }

    int judge2;
    printf("about update\n");
    printf("your jugde is 1 => SGD\n");
    printf("your jugde is 2 => AdaGrad\n");
    printf("your jugde is 3 => Momentum\n");
    printf("your judge is ");
    scanf("%d", &judge2);
    printf("\n");

    float *dEdA1ave = malloc(sizeof(float) * 784 * 50);
    float *dEdb1ave = malloc(sizeof(float) * 50);
    float *dEdA2ave = malloc(sizeof(float) * 50 * 100);
    float *dEdb2ave = malloc(sizeof(float) * 100);
    float *dEdA3ave = malloc(sizeof(float) * 100 * 10);
    float *dEdb3ave = malloc(sizeof(float) * 10);
    float *y1 = malloc(sizeof(float) * 50);
    float *y2 = malloc(sizeof(float) * 100);
    float *y3 = malloc(sizeof(float) * 10);
    float *dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *dEdb1 = malloc(sizeof(float) * 50);
    float *dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *dEdb2 = malloc(sizeof(float) * 100);
    float *dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *dEdb3 = malloc(sizeof(float) * 10);
    int *index = malloc(sizeof(int) * train_count);
    float *relu1x=malloc(sizeof(float)*50);
    float *relu2x=malloc(sizeof(float)*100);
    float *dEdx10=malloc(sizeof(float)*10);
    float *dEdx100 = malloc(sizeof(float) * 100);
    float *dEdx50 = malloc(sizeof(float) * 50);
    float *dEdx784 = malloc(sizeof(float) * 784);
    float *yre = malloc(sizeof(float) * 10);
    float *hA1 = malloc(sizeof(float) * 784 * 50);
    float *hb1 = malloc(sizeof(float) * 50);
    float *hA2 = malloc(sizeof(float) * 50 * 100);
    float *hb2 = malloc(sizeof(float) * 100);
    float *hA3 = malloc(sizeof(float) * 100 * 10);
    float *hb3 = malloc(sizeof(float) * 10);

    int epoc = 10;//エポック数
    int batch_size = 100;//バッチサイズ
    float sgd_learning_rate = 0.1;//SGDの学習係数
    float adagrad_learning_rate = 0.01;//adagradの学習係数
    float momentum_learning_rate = 0.01;//momentumの学習係数
    float momentum_parameter = 0.9;//momentumのハイパーパラメータα

    for (int j = 0; j < train_count;j++){
            index[j] = j;
        }
    
    for (int i = 0; i < epoc;i++){
        init(39200, 0, hA1);
        init(50, 0, hb1);
        init(5000, 0, hA2);
        init(100, 0, hb2);
        init(1000, 0, hA3);
        init(10, 0, hb3);
        printf("epoch%d\n", i+1);
        printf("                  0%%       50%%       100%%\n");
        printf("                  +---------+---------+\n");
        printf("        backward |");
        shuffle(train_count, index);
        
        for (int j = 0; j < train_count/batch_size;j++){
            init(39200, 0, dEdA1ave);
            init(50, 0, dEdb1ave);
            init(5000, 0, dEdA2ave);
            init(100, 0, dEdb2ave);
            init(1000, 0, dEdA3ave);
            init(10, 0, dEdb3ave);
            
            for (int k = 0; k < batch_size;k++){
                backward6(relu1x,relu2x,dEdx10,dEdx100,dEdx50,dEdx784,A1, b1, A2, b2, A3, b3, train_x + 784 * (index[100 * j + k]), train_y[index[100 * j + k]], y1, y2, y3, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3);//OK
                add(39200, dEdA1, dEdA1ave);
                add(5000, dEdA2, dEdA2ave);
                add(1000, dEdA3, dEdA3ave);
                add(50, dEdb1, dEdb1ave);
                add(100, dEdb2, dEdb2ave);
                add(10, dEdb3, dEdb3ave);
            }
            if(judge2 == 1){
                sgd(39200, sgd_learning_rate, dEdA1ave, A1);
                sgd(50, sgd_learning_rate, dEdb1ave, b1);
                sgd(5000, sgd_learning_rate, dEdA2ave, A2);
                sgd(100, sgd_learning_rate, dEdb2ave, b2);
                sgd(1000, sgd_learning_rate, dEdA3ave, A3);
                sgd(10, sgd_learning_rate, dEdb3ave, b3);
            }
            else if(judge2 == 2){
                adagrad(39200, adagrad_learning_rate, hA1, dEdA1ave, A1);
                adagrad(50, adagrad_learning_rate, hb1, dEdb1ave, b1);
                adagrad(5000, adagrad_learning_rate, hA2, dEdA2ave, A2);
                adagrad(100, adagrad_learning_rate, hb2, dEdb2ave, b2);
                adagrad(1000, adagrad_learning_rate, hA3, dEdA3ave, A3);
                adagrad(10, adagrad_learning_rate, hb3, dEdb3ave, b3);
            }
            else if(judge2 == 3){
                momentum(39200, momentum_learning_rate, momentum_parameter, dEdA1ave, A1, hA1);
                momentum(50, momentum_learning_rate, momentum_parameter, dEdb1ave, b1, hb1);
                momentum(5000, momentum_learning_rate, momentum_parameter, dEdA2ave, A2, hA2);
                momentum(100, momentum_learning_rate, momentum_parameter, dEdb2ave, b2, hb2);
                momentum(1000, momentum_learning_rate, momentum_parameter, dEdA3ave, A3, hA3);
                momentum(10, momentum_learning_rate, momentum_parameter, dEdb3ave, b3, hb3);
            }
            
            if(j%30==0){
                printf("#");
            }
        }
        printf("#");
        printf("\n");
        printf("test: train data |");
        int sum_train = 0;
        int sum_test = 0;
        float loss_train = 0;
        float loss_test = 0;
        for (int j = 0; j < train_count;j++){
            if(inference6(A1,b1,A2,b2,A3,b3,train_x+j*width*height,yre)==train_y[j]){
                sum_train++;
            }
            loss_train += cross_entropy_error(yre, train_y[j]);
            if(j%3000==0){
                printf("#");
            }
        }
        printf("#\n");
        printf("test:  test data |");
        for (int j = 0; j < test_count; j++){
            if (inference6(A1, b1, A2, b2, A3, b3, test_x + j * width * height, yre) == test_y[j]){
                sum_test++;
            }
            loss_test += cross_entropy_error(yre, test_y[j]);
            if(j%500==0){
                printf("#");
            }
        }
        printf("#\n");
        printf("train data\n");
        printf("correct answer rate : %f%%\n", sum_train * 100.0 / train_count);
        printf("loss : %f\n", loss_train);
        printf("test data\n");
        printf("correct answer rate : %f%%\n", sum_test * 100.0 / test_count);
        printf("loss : %f\n", loss_test);
        printf("\n");
    }

    save(argv[1], 784, 50, A1, b1);
    save(argv[2], 50, 100, A2, b2);
    save(argv[3], 100, 10, A3, b3);

    return 0;
}
