/* 기호 상수 */
const INPUT_NO = 2;
const HIDDEN_NO = 3;
const ALPHA = 29;
const SEED = 65536;
const MAX_INPUT_NO = 100;
const BIG_NUM = 100;
const LIMIT = 0.0001;

async function main() {
    const weightHiddenLayer: Array<Array<number>> = new Array(HIDDEN_NO).fill(1).map(() => new Array(INPUT_NO + 1).fill(0));
    const weightOutputLayer: Array<number> = new Array(HIDDEN_NO + 1).fill(0);
    const dataSet: Array<Array<number>> = new Array(MAX_INPUT_NO).fill(1).map(() => new Array(INPUT_NO + 1).fill(0));
    const outputHiddenLayer: Array<number> = new Array(HIDDEN_NO + 1).fill(0);
    let output: number;
    let error: number = BIG_NUM;
    let i, j;
    let countOfData;

    // 가중치 초기화
    initWeightHiddenLayer(weightHiddenLayer);
    initWeightOutputLayer(weightOutputLayer);
    printResult(weightHiddenLayer, weightOutputLayer);

    // 학습 데이터 읽기
    countOfData = await inputData(dataSet);
    console.log("학습 데이터 개수: ", countOfData);

    await wait(5000);

    // 학습
    while (error > LIMIT) {
        error = 0.0;
        for (j = 0; j < countOfData; j++) {
            // 순방향 계산
            output = forward(weightHiddenLayer, weightOutputLayer, outputHiddenLayer, dataSet[j]);
            // 출력층의 가중치 조정
            learnWeightOutputLayer(weightOutputLayer, outputHiddenLayer, dataSet[j], output);
            // 오차의 적산
            error += (output - dataSet[j][INPUT_NO]) * (output - dataSet[j][INPUT_NO]);
        }
        // 오차 출력
        // console.log(error);
    }
    // 학습 종료

    // 연결 강도 출력
    printResult(weightHiddenLayer, weightOutputLayer);

    // 학습 데이터에 대한 출력
    for (i = 0; i < countOfData; i++) {
        console.log(i, dataSet[i].join(" "), forward(weightHiddenLayer, weightOutputLayer, outputHiddenLayer, dataSet[i]));
    }

}

main();


async function wait(ms: number) {
    return new Promise(resolve => setTimeout(() => resolve(void 0), ms));
}

/**
 * 시그모이드 함수
 */
function sigmoid(u: number): number {
    return 1.0 / (1.0 + Math.exp(-u));
}

/**
 * 난수 생성
 * -1 (inclusive), and 1 (exclusive)
 */
function random(): number {
    return Math.random() * 2 - 1;
}

/**
 * 출력층 가중치 초기화
 * @param weightOutputLayer
 */
function initWeightOutputLayer(weightOutputLayer: Array<number>): void {
    for (let i = 0; i < HIDDEN_NO + 1; i++) {
        weightOutputLayer[i] = random();
    }
}

/**
 * 중간층 가중치 초기화
 * @param weightHiddenLayer
 */
function initWeightHiddenLayer(weightHiddenLayer: Array<Array<number>>): void {
    for (let i = 0; i < HIDDEN_NO; i++) {
        for (let j = 0; j < INPUT_NO + 1; j++) {
            weightHiddenLayer[i][j] = random();
        }
    }
    // weightHiddenLayer[0][0] = -2;
    // weightHiddenLayer[0][1] = 3;
    // weightHiddenLayer[0][2] = -1;
    // weightHiddenLayer[1][0] = -2;
    // weightHiddenLayer[1][1] = 1;
    // weightHiddenLayer[1][2] = 1;
}

/**
 * 결과 출력
 * @param weightHiddenLayer
 * @param weightOutputLayer
 */
function printResult(weightHiddenLayer: Array<Array<number>>, weightOutputLayer: Array<number>): void {
    console.log(weightHiddenLayer.map((arr: Array<number>) => arr.join(" ")).join(" "));
    console.log(weightOutputLayer.join(" "));
}

/**
 * 순방향 계산
 * @param weightHiddenLayer
 * @param weightOutputLayer
 * @param outputHiddenLayer
 * @param dataSet
 */
function forward(weightHiddenLayer: Array<Array<number>>, weightOutputLayer: Array<number>, outputHiddenLayer: Array<number>, dataSet: Array<number>): number {
    let i, j;
    let sumWeight: number;
    let result: number;

    // outputHiddenLayer 계산
    for (i = 0; i < HIDDEN_NO; i++) {
        sumWeight = 0;
        for (j = 0; j < INPUT_NO; j++) {
            sumWeight += dataSet[j] * weightHiddenLayer[i][j];
        }
        sumWeight -= weightHiddenLayer[i][j];
        outputHiddenLayer[i] = sigmoid(sumWeight);
    }

    // result 계산
    result = 0;
    for (let i = 0; i < HIDDEN_NO; i++) {
        result += outputHiddenLayer[i] * weightOutputLayer[i];
    }
    result -= weightOutputLayer[i];

    return sigmoid(result);
}

/**
 * 출력층 가중치 학습
 * @param weightOutputLayer
 * @param outputHiddenLayer
 * @param dataSet
 * @param output
 */
function learnWeightOutputLayer(weightOutputLayer: Array<number>, outputHiddenLayer: Array<number>, dataSet: Array<number>, output: number): void {
    let i;
    const d = (dataSet[INPUT_NO] - output) * output * (1 - output);
    for (i = 0; i < HIDDEN_NO; i++) {
        weightOutputLayer[i] += ALPHA * outputHiddenLayer[i] * d;
    }
    weightOutputLayer[i] += ALPHA * (-1.0) * d;
}

/**
 * 데이터 입력
 * @param dataSet
 */
async function inputData(dataSet: Array<Array<number>>): Promise<number> {
    return new Promise((resolve) => {
        let count = 0;
        process.stdin.resume();
        process.stdin.on("data", function (buf) {
            const rowData: Array<string> = buf.toString().split("\n");
            rowData.forEach((row) => {
                if (row.replace(/%s/g, "").length === 0) return;
                row.split(" ").forEach((d, i) => {
                    if (i > INPUT_NO || isNaN(parseInt(d))) return;
                    dataSet[count][i] = parseInt(d);
                });
                count++;
            });
        });
        process.stdin.on("end", function () {
            resolve(count);
        });
    });
}
