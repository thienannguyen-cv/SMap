// execute_ai_function.js
// Script này nhận dữ liệu JSON từ stdin, thực thi hàm bao AI và in kết quả ra stdout.

process.stdin.setEncoding('utf8');

let inputData = '';
process.stdin.on('data', (chunk) => {
    inputData += chunk;
});

process.stdin.on('end', () => {
    try {
        // payload từ Python:
        // {
        //   "fullJsFunctionString": "(function() { ... return executeGridPrediction; })()",
        //   "topLevelParamNames": ["rawSliceGrid", "targetGrid", "allConditionalArgs"],
        //   "rawSliceGrid": [[...]],
        //   "targetGrid": [[...]],
        //   "userGradientFlowsGrid": [[...]],
        //   "allConditionalArgs": { "conditionalBlockX": [[...]], ... }
        // }
        const { fullJsFunctionString, topLevelParamNames, rawSliceGrid, targetGrid, userGradientFlowsGrid, allConditionalArgs } = JSON.parse(inputData);

        // Bước 1: Eval chuỗi `fullJsFunctionString`.
        // Vì đây là một IIFE, việc eval nó sẽ trả về trực tiếp hàm `executeGridPrediction`.
        const executeGridPredictionFunction = eval(fullJsFunctionString);

        // Bước 2: Chuẩn bị các đối số thực tế cho hàm `executeGridPredictionFunction`.
        // Thứ tự của các đối số này phải khớp với `topLevelParamNames`.
        const actualArgsForGridPrediction = [];
        for (const paramName of topLevelParamNames) {
            if (paramName === "rawSliceGrid") {
                actualArgsForGridPrediction.push(rawSliceGrid);
            } else if (paramName === "targetGrid") {
                actualArgsForGridPrediction.push(targetGrid);
            } else if (paramName === "userGradientFlowsGrid") {
                actualArgsForGridPrediction.push(userGradientFlowsGrid);
            } else if (paramName === "allConditionalArgs") { // Tên của đối tượng conditionalArgs
                actualArgsForGridPrediction.push(allConditionalArgs);
            } else { // Fallback, không nên xảy ra nếu topLevelParamNames khớp
                actualArgsForGridPrediction.push(null); 
            }
        }

        // Bước 3: Thực thi hàm `executeGridPredictionFunction` với các đối số thực tế.
        const result = executeGridPredictionFunction(...actualArgsForGridPrediction);

        console.log(JSON.stringify({ status: 'SUCCESS', result: result }));

    } catch (e) {
        console.error(JSON.stringify({ status: 'ERROR', message: e.message, stack: e.stack }));
        process.exit(1);
    }
});