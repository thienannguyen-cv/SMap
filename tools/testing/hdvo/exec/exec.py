import argparse
import json
import numpy as np
import os
import sys
import re
import subprocess
import torch
import torch.nn.functional as F

# --- Hằng số của ứng dụng Fold & Unfold ---
FIXED_SLICES_COUNT = 9
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
HOTSPOT_SIZE = 8 # Kích thước cửa sổ vuông để tìm điểm nóng lỗi và là kích thước đầu ra

# --- Xác định thiết bị PyTorch
if torch.cuda.is_available():
    _global_torch_device = torch.device("cuda")
    print("PyTorch: CUDA is available. Using GPU.")
else:
    _global_torch_device = torch.device("cpu")
    print("PyTorch: CUDA is not available. Using CPU.")

# --- Cấu hình các trường dữ liệu mảng NPY/JSON và hình dạng mong muốn ---
NPY_JSON_FIELD_SHAPES = {
    "rawComponentSlidesData": {'dims': 4, 'slice_check': True}, # (channels, 9, H, W)
    "userDefinedGradientFlows": {'dims': 4, 'slice_check': True}, # (channels, 9, H, W)
    "targetGridData": {'dims': 3, 'slice_check': False}, # (channels, H, W)
    "inputGridData": {'dims': 3, 'slice_check': False}, # (channels, H, W)
    "conditionalBlockX": {'dims': 4, 'slice_check': True}, # Giả định conditionalBlockX là 4D
}

# --- Các trường metadata không phải mảng lưới ---
NON_GRID_METADATA_LIST_FIELDS = [
    "detailedMismatches",
    "mismatchHistory",
    "aiTheoryHistory",
    "aiPrognosis",
]

# --- Ngoại lệ tùy chỉnh ---
class SessionValidationError(Exception):
    """Ngoại lệ tùy chỉnh cho các lỗi xác thực dữ liệu hoặc phiên."""
    pass

# --- Hàm tiện ích tính toán kích thước (không thay đổi) ---
def calculate_output_size(input_size, kernel_size, stride, padding):
    return int(np.floor((input_size + 2 * padding - kernel_size) / stride) + 1)

def calculate_input_size(output_size, kernel_size, stride, padding):
    return int((output_size - 1) * stride - 2 * padding + kernel_size)

def get_actual_grid_shape(arr_shape, data_key):
    """Xác định hình dạng (channels, height, width) từ hình dạng NumPy/JSON."""
    if data_key in ["rawComponentSlidesData", "userDefinedGradientFlows", "conditionalBlockX"] or data_key.startswith("conditionalBlock_") or data_key.startswith("list_npy_item_") or data_key.startswith("list_json_array_item_"):
        if len(arr_shape) == 4:
            return (arr_shape[0], arr_shape[2], arr_shape[3])
    elif data_key in ["targetGridData", "inputGridData"]:
        if len(arr_shape) == 3:
            return (arr_shape[0], arr_shape[1], arr_shape[2])
    
    if len(arr_shape) == 4:
        return (arr_shape[0], arr_shape[2], arr_shape[3])
    elif len(arr_shape) == 3:
        return (arr_shape[0], arr_shape[1], arr_shape[2])
    
    return None

# Hàm xác thực và chuyển đổi trường mảng NPY/JSON (Điều chỉnh để chấp nhận hình chữ nhật và kiểm tra nhất quán) ---
def validate_and_convert_array_field(array_value, data_key, session_dimensions, input_session_data_root_path):
    is_npy_file = isinstance(array_value, str) and array_value.endswith('.npy')
    
    file_info = {
        'original_value': array_value,
        'json_field': data_key,
        'shape': None,
        'status': 'SUCCESS',
        'error_message': None
    }

    try:
        if is_npy_file:
            npy_path = os.path.join("./", array_value)
            file_info['file_path'] = npy_path
            if not os.path.exists(npy_path):
                raise SessionValidationError(f"Tệp NumPy '{array_value}' không tìm thấy.")
            arr = np.load(npy_path)
            file_info['shape'] = arr.shape
        elif isinstance(array_value, list):
            try:
                arr = np.array(array_value)
                file_info['shape'] = arr.shape
            except Exception as e:
                raise SessionValidationError(f"Dữ liệu JSON tường minh không thể chuyển đổi thành mảng NumPy hợp lệ. Chi tiết: {e}")
        else:
            raise SessionValidationError(f"Giá trị trường '{data_key}' không phải đường dẫn .npy hoặc mảng JSON hợp lệ.")

        current_actual_grid_shape = get_actual_grid_shape(arr.shape, data_key)

        if current_actual_grid_shape is None:
            raise SessionValidationError(f"Hình dạng '{arr.shape}' không phải 3D hoặc 4D hợp lệ cho dữ liệu lưới.")

        expected_dims_config = NPY_JSON_FIELD_SHAPES.get(data_key)
        if expected_dims_config:
            if len(arr.shape) != expected_dims_config['dims']:
                raise SessionValidationError(f"Số chiều không hợp lệ. Cần {expected_dims_config['dims']} chiều, nhận được {len(arr.shape)}.")
            if expected_dims_config['slice_check'] and arr.shape[1] != FIXED_SLICES_COUNT:
                raise SessionValidationError(f"Số tấm không hợp lệ. Cần {FIXED_SLICES_COUNT} tấm, nhận được {arr.shape[1]}.")

        # Kiểm tra tính nhất quán cho cả height và width riêng biệt
        if session_dimensions['channels'] is None:
            session_dimensions['channels'] = current_actual_grid_shape[0]
            session_dimensions['height'] = current_actual_grid_shape[1]
            session_dimensions['width'] = current_actual_grid_shape[2]
        else:
            if current_actual_grid_shape[0] != session_dimensions['channels'] or \
               current_actual_grid_shape[1] != session_dimensions['height'] or \
               current_actual_grid_shape[2] != session_dimensions['width']:
                raise SessionValidationError(
                    f"Kích thước mảng ({current_actual_grid_shape}) không khớp với kích thước phiên chung đã xác định "
                    f"(kênh: {session_dimensions['channels']}, cao: {session_dimensions['height']}, rộng: {session_dimensions['width']})."
                )

        return arr.tolist(), file_info

    except SessionValidationError as e:
        file_info['status'] = 'ERROR'
        file_info['error_message'] = f"Lỗi xác thực: {e}"
        return None, file_info
    except Exception as e:
        file_info['status'] = 'ERROR'
        file_info['error_message'] = f"Lỗi xử lý mảng: {e}"
        return None, file_info


# Hàm duyệt JSON đệ quy (Không thay đổi) ---
def traverse_and_process_arrays(data, session_dimensions, processed_files_info, input_session_data_root_path):
    """
    Duyệt đệ quy qua cấu trúc dữ liệu JSON và xử lý các trường mảng.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in NON_GRID_METADATA_LIST_FIELDS:
                if isinstance(value, (dict, list)):
                    traverse_and_process_arrays(value, session_dimensions, processed_files_info, input_session_data_root_path)
                continue

            if key in NPY_JSON_FIELD_SHAPES or (isinstance(value, str) and value.endswith('.npy')):
                converted_data, info = validate_and_convert_array_field(value, key, session_dimensions, input_session_data_root_path)
                processed_files_info.append(info)
                if info['status'] == 'SUCCESS':
                    data[key] = converted_data
                else:
                    data[key] = None
            elif isinstance(value, list) and not key.startswith("conditionalBlock"):
                converted_data, info = validate_and_convert_array_field(value, key, session_dimensions, input_session_data_root_path)
                processed_files_info.append(info)
                if info['status'] == 'SUCCESS':
                    data[key] = converted_data
                else:
                    data[key] = None
            elif isinstance(value, dict):
                traverse_and_process_arrays(value, session_dimensions, processed_files_info, input_session_data_root_path)
            elif key.startswith("conditionalBlock") and isinstance(value, (str, list)):
                converted_data, info = validate_and_convert_array_field(value, key, session_dimensions, input_session_data_root_path)
                processed_files_info.append(info)
                if info['status'] == 'SUCCESS':
                    data[key] = converted_data
                else:
                    data[key] = None

    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str) and item.endswith('.npy'):
                converted_data, info = validate_and_convert_array_field(item, f"list_npy_item_{i}", session_dimensions, input_session_data_root_path)
                processed_files_info.append(info)
                if info['status'] == 'SUCCESS':
                    data[i] = converted_data
                else:
                    data[i] = None
            elif isinstance(item, list):
                converted_data, info = validate_and_convert_array_field(item, f"list_json_array_item_{i}", session_dimensions, input_session_data_root_path)
                processed_files_info.append(info)
                if info['status'] == 'SUCCESS':
                    data[i] = converted_data
                else:
                    data[i] = None
            elif isinstance(item, dict):
                traverse_and_process_arrays(item, session_dimensions, processed_files_info, input_session_data_root_path)

# --- KHỐI MÃ ĐƯỢC THAY ĐỔI / THÊM MỚI ---
def _generate_js_wrapper_for_grid_prediction(user_ai_function_body_str, conditional_block_names, num_channels, grid_height, grid_width, comparison_mode, tolerance_epsilon=0):
    """
    Tạo một chuỗi JavaScript chứa định nghĩa của hàm AI gốc và một hàm bao
    để xử lý toàn bộ lưới, trả về hàm bao đó.
    """
    # Trích xuất tham số của hàm AI gốc của người dùng
    param_match = re.search(r'function\s*(\w*)\s*\(([^)]*)\)', user_ai_function_body_str) # Capture function name
    user_ai_param_names_for_inner_call = []
    user_func_name = "__user_ai_logic__" # Default internal name
    
    if param_match:
        if param_match.group(1): # If function has a name
            user_func_name = param_match.group(1) # Use original name, then rename later
        user_ai_param_names_for_inner_call = [p.strip() for p in param_match.group(2).split(',') if p.strip()]
    else:
        # Fallback if user only provides function body without `function` keyword
        user_ai_param_names_for_inner_call = ["allChannelsSliceValuesAtOutputCell", "allChannelsTargetValuesAtOutputCell"]
        user_ai_param_names_for_inner_call.extend(conditional_block_names)
    
    # Bọc body AI của người dùng vào một hàm có tên nội bộ __user_ai_logic__
    inner_user_ai_function_definition = f"const __user_ai_logic__ = function({', '.join(user_ai_param_names_for_inner_call)}) {{ {user_ai_function_body_str} }};"

    # Định nghĩa các tham số cho hàm bao chính (nhận toàn bộ lưới)
    wrapper_function_params = [
        "rawSliceGrid",              # (C, S, H, W)
        "targetGrid",                # (C, H, W)
        "userGradientFlowsGrid",     # (C, S, H, W)
        "allConditionalArgs"         # Object containing all conditional blocks by name
    ]

    # Xây dựng chuỗi hàm JavaScript đầy đủ, trả về hàm `executeGridPrediction`
    # Đây là một IIFE (Immediately Invoked Function Expression)
    full_js_string = f"""
    (function() {{
        {inner_user_ai_function_definition} // Định nghĩa hàm AI gốc của người dùng

        function executeGridPrediction({', '.join(wrapper_function_params)}) {{
            const NUM_CHANNELS = {num_channels};
            const HEIGHT = {grid_height};
            const WIDTH = {grid_width};
            const COMPARISON_MODE = '{comparison_mode}';
            const TOLERANCE_EPS = {tolerance_epsilon};
            const FIXED_SLICES = {FIXED_SLICES_COUNT};

            const errorGrid = Array.from({{ length: NUM_CHANNELS }}, () => Array.from({{ length: HEIGHT }}, () => Array(WIDTH).fill(0)));

            for (let l = 0; l < NUM_CHANNELS; l++) {{
                for (let r = 0; r < HEIGHT; r++) {{
                    for (let c = 0; c < WIDTH; c++) {{
                        let totalErrorsInCell = 0;

                        const perCellSliceData = Array.from({{ length: 1 }}, (_, ch_idx) => {{
                            return Array.from({{ length: FIXED_SLICES }}, (_, s_idx) => rawSliceGrid[l][s_idx][r][c]);
                        }});

                        const perCellTargetData = Array.from({{ length: 1 }}, (_, ch_idx) => {{
                            return targetGrid[l][r][c];
                        }});
                        
                        const perUserGradientFlowsData = Array.from({{ length: 1 }}, (_, ch_idx) => {{
                            return Array.from({{ length: FIXED_SLICES }}, (_, s_idx) => userGradientFlowsGrid[l][s_idx][r][c]);
                        }});

                        const perCellConditionalArgsForInnerAI = {{}};
                        for (const cbName of {json.dumps(conditional_block_names)}) {{
                            const cbGrid = allConditionalArgs[cbName]; // Truy cập conditional block từ object được truyền vào
                            if (cbGrid && cbGrid.length > 0) {{
                                // Check if it's 4D or 3D or 2D (C,S) or 1D (C,)
                                if (Array.isArray(cbGrid[0]) && Array.isArray(cbGrid[0][0]) && cbGrid[0].length === FIXED_SLICES) {{ // 4D (C,S,H,W)
                                    perCellConditionalArgsForInnerAI[cbName] = Array.from({{ length: 1 }}, (_, ch_idx) => {{
                                        return Array.from({{ length: FIXED_SLICES }}, (_, s_idx) => cbGrid[l][s_idx][r][c]);
                                    }});
                                }} else if (Array.isArray(cbGrid[0]) && !Array.isArray(cbGrid[0][0])) {{ // 3D (C,H,W) or 2D (C,S)
                                    // Need to differentiate between (C,H,W) and (C,S) if 'H' can be 9
                                    // Assuming (C,H,W) for this block
                                    perCellConditionalArgsForInnerAI[cbName] = Array.from({{ length: 1 }}, (_, ch_idx) => {{
                                        return cbGrid[l][r][c];
                                    }});
                                }} else {{ // If not an array of arrays, assume it's (C,) scalar values per channel
                                    perCellConditionalArgsForInnerAI[cbName] = Array.from({{ length: 1 }}, (_, ch_idx) => {{
                                        return cbGrid[l]; 
                                    }});
                                }}
                            }}
                        }}
                        
                        let aiPredictionResult;
                        try {{
                            const userFuncArgs = [];
                            for (const paramName of {json.dumps(user_ai_param_names_for_inner_call)}) {{
                                if (paramName === "allChannelsSliceValuesAtOutputCell") {{
                                    userFuncArgs.push(perCellSliceData);
                                }} else if (paramName === "allChannelsTargetValuesAtOutputCell") {{
                                    userFuncArgs.push(perCellTargetData);
                                }} else if (perCellConditionalArgsForInnerAI[paramName] !== undefined) {{
                                    userFuncArgs.push(perCellConditionalArgsForInnerAI[paramName]);
                                }} else {{
                                    userFuncArgs.push(null); 
                                }}
                            }}
                            aiPredictionResult = __user_ai_logic__(...userFuncArgs); // Gọi logic AI của người dùng
                        }} catch (e) {{
                            console.error('Error in user AI logic for cell (' + r + ',' + c + '): ' + e.message + '\\nStack: ' + e.stack);
                            totalErrorsInCell = 1 * FIXED_SLICES;
                            errorGrid[l][r][c] = totalErrorsInCell;
                            continue; 
                        }}

                        if (Array.isArray(aiPredictionResult) && aiPredictionResult.length === 1) {{
                            for (let ch_idx = 0; ch_idx < 1; ch_idx++) {{
                                if (Array.isArray(aiPredictionResult[ch_idx]) && aiPredictionResult[ch_idx].length === FIXED_SLICES) {{
                                    for (let s_idx = 0; s_idx < FIXED_SLICES; s_idx++) {{
                                        let currentFlowValue = perUserGradientFlowsData[ch_idx][s_idx];
                                        if (COMPARISON_MODE === 'sign') {{
                                            currentFlowValue = Math.sign(currentFlowValue);
                                            
                                        }}
                                        if (Math.abs(aiPredictionResult[ch_idx][s_idx] - currentFlowValue) > TOLERANCE_EPS) {{
                                            totalErrorsInCell++;
                                        }}
                                        
                                    }}
                                }} else {{
                                    console.warn('Invalid AI prediction shape for channel ' + l + ' in cell (' + r + ',' + c + '). Expected [' + FIXED_SLICES + ']. Got: ' + JSON.stringify(aiPredictionResult[ch_idx]));
                                    totalErrorsInCell += FIXED_SLICES;
                                }}
                            }}
                        }} else {{
                            console.error('Invalid AI prediction result shape for cell (' + r + ',' + c + '). Expected [' + 1 + '][' + FIXED_SLICES + ']. Got: ' + JSON.stringify(aiPredictionResult));
                            totalErrorsInCell = 1 * FIXED_SLICES;
                        }}
                        
                        errorGrid[l][r][c] = totalErrorsInCell;
                    }}
                }}
            }}
            return errorGrid;
        }}
        return executeGridPrediction; // IIFE trả về hàm executeGridPrediction
    }})() // IIFE được thực thi ngay lập tức
    """
    # Trả về chuỗi JS chứa IIFE và danh sách tên tham số mà executeGridPrediction mong đợi
    return full_js_string, wrapper_function_params

# Hàm thực thi JavaScript AI Function (Đã sửa để gọi hàm bao mới)
def _execute_js_ai_function_grid_level(user_ai_function_body_str, raw_slides_grid_data, target_grid_data, user_gradient_flows_grid_data, conditional_blocks_grid_data_map, num_channels, grid_height, grid_width, comparison_mode, tolerance_epsilon):
    try:
        # Tạo hàm bao JS đầy đủ (IIFE) và nhận danh sách tên tham số cấp cao của hàm bao
        full_js_string_to_eval, top_level_param_names = _generate_js_wrapper_for_grid_prediction(
            user_ai_function_body_str,
            list(conditional_blocks_grid_data_map.keys()),
            num_channels,
            grid_height,
            grid_width,
            comparison_mode, 
            tolerance_epsilon
        )

        payload = {
            "fullJsFunctionString": full_js_string_to_eval, # Đây là chuỗi JS IIFE
            "topLevelParamNames": top_level_param_names, # Danh sách tên tham số mà executeGridPrediction mong đợi
            "rawSliceGrid": raw_slides_grid_data,
            "targetGrid": target_grid_data,
            "userGradientFlowsGrid": user_gradient_flows_grid_data,
            "allConditionalArgs": conditional_blocks_grid_data_map # Đối tượng chứa tất cả conditional blocks
        }
        json_payload = json.dumps(payload)

        node_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'execute_ai_function.js')

        result = subprocess.run(
            ['node', node_script_path],
            input=json_payload,
            capture_output=True,
            text=True,
            check=True
        )

        node_output = json.loads(result.stdout)

        if node_output.get('status') == 'SUCCESS':
            return node_output.get('result')
        else:
            raise SessionValidationError(f"Lỗi từ Node.js: {node_output.get('message', 'Không rõ lỗi')}\nStack: {node_output.get('stack', 'N/A')}")

    except subprocess.CalledProcessError as e:
        raise SessionValidationError(f"Lỗi khi gọi Node.js subprocess: {e.stderr.strip() if e.stderr else 'Không có thông báo lỗi từ Node.js'}")
    except json.JSONDecodeError as e:
        stdout_content = result.stdout if 'result' in locals() and result.stdout else 'Không có đầu ra'
        raise SessionValidationError(f"Lỗi phân tích JSON từ Node.js: {e}. Output Node.js: {stdout_content}")
    except FileNotFoundError:
        raise SessionValidationError("Lỗi: Không tìm thấy lệnh 'node' hoặc file 'execute_ai_function.js'. Vui lòng đảm bảo Node.js được cài đặt và 'execute_ai_function.js' nằm trong cùng thư mục với script Python này.")
    except Exception as e:
        raise SessionValidationError(f"Lỗi không mong muốn khi thực thi JS qua Node.js: {e}")

def _crop_session_data_to_hotspot(session_data, ch_idx, r_start, c_start, crop_size, original_num_channels, original_height, original_width):
    """
    Cắt tất cả các mảng chính trong session_data xuống kích thước hotspot (luôn là 14x14).
    """
    cropped_session_data = session_data.copy()
    
    # Cập nhật metadata phiên trong file đầu ra để luôn là kích thước hotspot
    cropped_session_data['numChannels'] = 1 # Số kênh vẫn giữ nguyên
    cropped_session_data['outputGridSize'] = crop_size # Luôn là 14 (HOTSPOT_SIZE)
    cropped_session_data['inputSize'] = calculate_input_size(crop_size, KERNEL_SIZE, STRIDE, PADDING)
    cropped_session_data['rStart'] = original_height-r_start
    cropped_session_data['cStart'] = c_start
    if 'outputGridWidth' in cropped_session_data:
        del cropped_session_data['outputGridWidth']

    arrays_to_crop = [
        "rawComponentSlidesData",
        "targetGridData",
        "userDefinedGradientFlows",
        "inputGridData"
    ]
    
    for key in list(cropped_session_data.keys()):
        if key.startswith("conditionalBlock") and isinstance(cropped_session_data[key], list) and key not in NON_GRID_METADATA_LIST_FIELDS:
            arrays_to_crop.append(key)

    for array_key in arrays_to_crop:
        if array_key in cropped_session_data and isinstance(cropped_session_data[array_key], list):
            original_array_np = np.array(cropped_session_data[array_key])
            
            if len(original_array_np.shape) == 4 and original_array_np.shape[2] >= r_start + crop_size and original_array_np.shape[3] >= c_start + crop_size:
                cropped_array = original_array_np[ch_idx:(ch_idx+1), :, r_start:r_start+crop_size, c_start:c_start+crop_size].tolist()
                cropped_session_data[array_key] = cropped_array
            elif len(original_array_np.shape) == 3 and original_array_np.shape[1] >= r_start + crop_size and original_array_np.shape[2] >= c_start + crop_size:
                cropped_array = original_array_np[ch_idx:(ch_idx+1), r_start:r_start+crop_size, c_start:c_start+crop_size].tolist()
                cropped_session_data[array_key] = cropped_array
            else:
                print(f"Cảnh báo: Không thể cắt mảng '{array_key}' từ kích thước {original_array_np.shape} vì hotspot nằm ngoài. Giữ nguyên.")
    
    return cropped_session_data

# ... (rest of process_session_file and other functions remain the same as previous response) ...
def process_session_file(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp JSON đầu vào tại đường dẫn: {input_path}")
        
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Lỗi: Tệp JSON đầu vào không hợp lệ. Chi tiết: {e}")

    input_session_data_root_path = input_path

    session_dimensions = {
        'channels': None,
        'height': None,
        'width': None
    }

    processed_files_info = []
    accumulated_errors = []
    
    try:
        traverse_and_process_arrays(session_data, session_dimensions, processed_files_info, input_session_data_root_path)
    except Exception as e:
        accumulated_errors.append(f"Lỗi không mong muốn trong quá trình duyệt cấu trúc JSON: {e}")

    print("\n--- Báo cáo Xử lý Tệp Mảng ---")
    if not processed_files_info:
        print("Không tìm thấy tệp NumPy hoặc mảng JSON tường minh nào để xử lý trong tệp JSON đầu vào.")

    for info in processed_files_info:
        if info['status'] == 'ERROR':
            accumulated_errors.append(f"Lỗi ở trường '{info['json_field']}' (giá trị gốc: '{info['original_value'] if isinstance(info['original_value'], str) else 'JSON Array'}'): {info['error_message']}")
        
        if info['status'] == 'SUCCESS':
            print(f"✅ Thành công: '{info['json_field']}' (giá trị gốc: '{info['original_value'] if isinstance(info['original_value'], str) else 'JSON Array'}')")
            print(f"   Hình dạng (Shape): {info['shape']}")
        else:
            print(f"❌ Lỗi: '{info['json_field']}' (giá trị gốc: '{info['original_value'] if isinstance(info['original_value'], str) else 'JSON Array'}')")
            print(f"   Chi tiết: {info['error_message']}")

    if session_dimensions['channels'] is not None and \
       session_dimensions['height'] is not None and \
       session_dimensions['width'] is not None:
        
        session_data['numChannels'] = session_dimensions['channels']
        session_data['outputGridSize'] = session_dimensions['height']
        session_data['outputGridWidth'] = session_dimensions['width']
        
        calculated_input_size = calculate_input_size(
            session_dimensions['height'], KERNEL_SIZE, STRIDE, PADDING
        )
        if calculated_input_size < 1:
            error_msg = f"Lỗi: Kích thước đầu vào suy ra ({calculated_input_size}) không hợp lệ. Vui lòng kiểm tra lại kích thước đầu ra hoặc thông số kernel/stride/padding."
            accumulated_errors.append(error_msg)
        session_data['inputSize'] = calculated_input_size
    else:
        if not accumulated_errors:
             accumulated_errors.append("Lỗi: Không thể suy ra kích thước 'numChannels', 'inputSize', 'outputGridSize' từ các tệp mảng. "
                                      "Đảm bảo có ít nhất một trường mảng chính hợp lệ và các mảng đó có hình dạng hợp lệ.")

    print(f"\n--- Tính toán Lưới điểm lỗi và Tổng số lỗi Toàn cục ---")
    global_total_errors = 0
    cell_error_scores_grid = None

    try:
        raw_slides_np = np.array(session_data.get("rawComponentSlidesData")) if session_data.get("rawComponentSlidesData") is not None else None
        target_grid_np = np.array(session_data.get("targetGridData")) if session_data.get("targetGridData") is not None else None
        user_gradient_flows_np = np.array(session_data.get("userDefinedGradientFlows")) if session_data.get("userDefinedGradientFlows") is not None else None

        
        print("Mảng raw_slides_np")
        print((raw_slides_np.min(), raw_slides_np.max()))
        print((raw_slides_np.argmin(), raw_slides_np.argmax()))
        
        print("Mảng target_grid_np")
        print((target_grid_np.min(), target_grid_np.max()))
        print((target_grid_np.argmin(), target_grid_np.argmax()))
        
        print("Mảng user_gradient_flows_np")
        print((user_gradient_flows_np.min(), user_gradient_flows_np.max()))
        print((user_gradient_flows_np.argmin(), user_gradient_flows_np.argmax()))
            
        if raw_slides_np is None or target_grid_np is None or not raw_slides_np.size or not target_grid_np.size:
            raise SessionValidationError("Không có dữ liệu rawComponentSlidesData hoặc targetGridData hợp lệ để tính toán lỗi toàn cục.")

        # Lấy kích thước thực tế của lưới từ session_dimensions
        current_output_grid_height = session_dimensions['height']
        current_output_grid_width = session_dimensions['width']
        current_num_channels = session_dimensions['channels']
        current_comparison_mode, current_tolerance_epsilon = session_data['comparisonMode'], session_data['toleranceEpsilon']

        ai_function_body = session_data.get("aiFunctionBody")
        if not ai_function_body:
            raise SessionValidationError("aiFunctionBody không có trong dữ liệu phiên. Không thể tính toán lỗi toàn cục.")

        # Thu thập conditional blocks dưới dạng NumPy arrays
        conditional_blocks_np_map = {}
        for key in session_data.keys():
            if key.startswith("conditionalBlock") and isinstance(session_data[key], list) and key not in NON_GRID_METADATA_LIST_FIELDS:
                try:
                    conditional_blocks_np_map[key] = np.array(session_data[key])
                except Exception as e:
                    accumulated_errors.append(f"Cảnh báo: Không thể chuyển đổi conditionalBlock '{key}' sang NumPy: {e}")


        try:
            # GỌI HÀM AI MỘT LẦN VỚI TOÀN BỘ LƯỚI (thông qua hàm bao)
            error_grid_from_js = _execute_js_ai_function_grid_level(
                ai_function_body,
                raw_slides_np.tolist(),
                target_grid_np.tolist(),
                user_gradient_flows_np.tolist(),
                {name: arr.tolist() for name, arr in conditional_blocks_np_map.items()},
                current_num_channels,
                current_output_grid_height,
                current_output_grid_width,
                current_comparison_mode, current_tolerance_epsilon
            )

            cell_error_scores_grid = np.array(error_grid_from_js)
            global_total_errors = np.sum(cell_error_scores_grid)
        except Exception as e:
            accumulated_errors.append(f"Lỗi khi thực thi hàm AI cho toàn bộ lưới: {e}")
            raise

        print(f"Tổng số lỗi toàn cục trên lưới: {global_total_errors}.")

    except SessionValidationError as e:
        accumulated_errors.append(f"Lỗi khi tính toán tổng số lỗi toàn cục: {e}")
    except Exception as e:
        accumulated_errors.append(f"Lỗi không mong muốn khi tính toán tổng số lỗi toàn cục: {e}")

    # Phần tìm hotspot và cắt dữ liệu bằng PyTorch (Đã điều chỉnh logic hotspot cho hình chữ nhật) ---
    if cell_error_scores_grid is not None and \
       (current_output_grid_height >= HOTSPOT_SIZE and current_output_grid_width >= HOTSPOT_SIZE):

        print(f"\n--- Tìm kiếm Hotspot Lỗi {HOTSPOT_SIZE}x{HOTSPOT_SIZE} bằng PyTorch Convolution ---")
        try:
            # Chuyển lưới điểm lỗi sang tensor PyTorch và đưa lên thiết bị
            error_grid_tensor = torch.tensor(cell_error_scores_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_global_torch_device)
            temp_error_grid_tensor = error_grid_tensor.reshape(current_num_channels,-1)
            max_ch_idx = torch.argmax(torch.sum(temp_error_grid_tensor,dim=-1,keepdim=False))
            error_grid_tensor = error_grid_tensor[max_ch_idx, ...]

            # Tạo kernel (bộ lọc) để tính tổng trong cửa sổ HOTSPOT_SIZE x HOTSPOT_SIZE
            kernel_tensor = torch.ones((1, 1, HOTSPOT_SIZE, HOTSPOT_SIZE), dtype=torch.float32, device=_global_torch_device)

            # Thực hiện tích chập
            summed_windows = F.conv2d(error_grid_tensor, kernel_tensor, stride=1, padding=0)

            # Tìm giá trị lớn nhất và vị trí của nó
            max_errors_tensor = torch.max(summed_windows)
            
            max_idx = torch.argmax(summed_windows)
            target_max_abs_tensor = 0
            if max_errors_tensor.item() == 0:
                target_max_abs_tensor = F.conv2d(torch.tensor(torch.from_numpy(np.abs(target_grid_np)).reshape(*cell_error_scores_grid.shape), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_global_torch_device), kernel_tensor, stride=1, padding=0)
                max_idx = torch.argmax(target_max_abs_tensor)

            output_height_conv = current_output_grid_height - HOTSPOT_SIZE + 1
            output_width_conv = current_output_grid_width - HOTSPOT_SIZE + 1

            best_r_start = max_idx // output_width_conv
            best_c_start = max_idx % output_width_conv

            max_errors = max_errors_tensor.item()
            best_hotspot_coords = (max_ch_idx.item(), best_r_start.item(), best_c_start.item())

            print(f"✅ Đã tìm thấy hotspot lỗi lớn nhất tại vị trí (kenh: {best_hotspot_coords[0]}, hàng: {best_hotspot_coords[1]}, cột: {best_hotspot_coords[2]}) với tổng số lỗi: {max_errors}.") 
            
            session_data = _crop_session_data_to_hotspot(session_data, best_hotspot_coords[0], best_hotspot_coords[1], best_hotspot_coords[2], HOTSPOT_SIZE, current_num_channels, current_output_grid_height, current_output_grid_width)
            print(f"Đã cắt dữ liệu phiên xuống kích thước {HOTSPOT_SIZE}x{HOTSPOT_SIZE}.")

        except Exception as e:
            accumulated_errors.append(f"Lỗi khi thực hiện phân tích hotspot bằng PyTorch: {e}")
            print("Không tìm thấy hotspot lỗi (hoặc lỗi trong quá trình xử lý PyTorch). Dữ liệu không bị cắt.")
    else:
        print(f"\nKích thước lưới đầu ra ({session_dimensions.get('height')}x{session_dimensions.get('width')}) không đủ lớn để tìm hotspot {HOTSPOT_SIZE}x{HOTSPOT_SIZE}.")
        print("Không thực hiện tìm kiếm hotspot và cắt dữ liệu.")

    # Bước 5: Kiểm tra các trường bắt buộc khác và các giá trị mặc định (không thay đổi) ---
    # Các trường này được đảm bảo có trong session_data trước khi ghi file
    session_data.setdefault("aiFunctionBody", "function predictGradientFlows(allChannelsSliceValuesAtOutputCell, allChannelsTargetValuesAtOutputCell){let output = Array.from({ length: allChannelsSliceValuesAtOutputCell.length }, (_, ch) => Array.from({ length: 9 }, () => 0)); return output;}")
    session_data.setdefault("lastValidAiFunctionBody", None)
    session_data.setdefault("aiNaturalLanguageExplanation", "")
    session_data.setdefault("aiTheory", "")
    session_data.setdefault("aiIterationCount", 0)
    session_data.setdefault("userHint", "")
    session_data.setdefault("promptMode", "high-end")
    session_data.setdefault("mismatchHistory", [])
    session_data.setdefault("aiTheoryHistory", [])
    session_data.setdefault("isAISuggestingHint", False)
    session_data.setdefault("aiPrognosis", [])
    session_data.setdefault("isChannelSyncEnabled", False)
    session_data.setdefault("selectedComparisonSource", "output")

    # --- Bước 6: Kết luận và ghi tệp ---
    if not accumulated_errors:
        print("\n--- Kết quả Hoàn tất ---")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=4)
            print(f"Đã chuyển đổi thành công. Dữ liệu phiên được lưu vào: {output_path}")
        except IOError as e:
            raise IOError(f"Lỗi: Không thể ghi tệp đầu ra tại đường dẫn: {output_path}. "
                          f"Vui lòng kiểm tra quyền truy cập thư mục. Chi tiết: {e}")
    else:
        print("\n--- Tóm tắt Lỗi Tổng thể ---")
        for err_msg in accumulated_errors:
            print(f"❌ {err_msg}")
        print("\nKhông thể tạo tệp đầu ra do có lỗi trong quá trình xử lý hoặc xác thực.")
        sys.exit(1)

# --- HÀM MỚI để xử lý nhiều tệp ---
def process_multiple_sessions(input_paths, output_path):
    # Đọc tệp đầu tiên để lấy dữ liệu lưới chính
    first_input_path = input_paths[0]
    if not os.path.exists(first_input_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp JSON đầu vào đầu tiên tại đường dẫn: {first_input_path}")
    
    try:
        with open(first_input_path, 'r', encoding='utf-8') as f:
            base_session_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Lỗi: Tệp JSON đầu vào '{first_input_path}' không hợp lệ. Chi tiết: {e}")

    # Xử lý các mảng và xác định kích thước từ tệp đầu tiên
    session_dimensions = {'channels': None, 'height': None, 'width': None}
    processed_files_info = []
    accumulated_errors = []

    try:
        traverse_and_process_arrays(base_session_data, session_dimensions, processed_files_info, first_input_path)
    except Exception as e:
        accumulated_errors.append(f"Lỗi không mong muốn trong quá trình duyệt cấu trúc JSON của tệp chính: {e}")

    # Kiểm tra kích thước lưới sau khi xử lý
    if not session_dimensions['height'] or not session_dimensions['width']:
        accumulated_errors.append("Lỗi: Không thể suy ra kích thước lưới từ tệp đầu tiên. Vui lòng kiểm tra lại dữ liệu.")
        print("\n--- Tóm tắt Lỗi Tổng thể ---")
        for err_msg in accumulated_errors:
            print(f"❌ {err_msg}")
        sys.exit(1)

    # Thu thập tất cả aiFunctionBody từ tất cả các tệp đầu vào
    ai_function_bodies = []
    for input_path in input_paths:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "aiFunctionBody" in data:
                    ai_function_bodies.append(data["aiFunctionBody"])
                else:
                    print(f"Cảnh báo: Tệp '{input_path}' không chứa trường 'aiFunctionBody'. Bỏ qua.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            accumulated_errors.append(f"Lỗi khi đọc tệp '{input_path}' để lấy 'aiFunctionBody': {e}")
            
    if not ai_function_bodies:
        accumulated_errors.append("Không tìm thấy aiFunctionBody nào trong các tệp đầu vào.")
        print("\n--- Tóm tắt Lỗi Tổng thể ---")
        for err_msg in accumulated_errors:
            print(f"❌ {err_msg}")
        sys.exit(1)

    # Lấy dữ liệu lưới chính từ tệp đầu tiên đã xử lý
    raw_slides_np = np.array(base_session_data.get("rawComponentSlidesData"))
    target_grid_np = np.array(base_session_data.get("targetGridData"))
    user_gradient_flows_np = np.array(base_session_data.get("userDefinedGradientFlows"))
    conditional_blocks_np_map = {
        key: np.array(value) for key, value in base_session_data.items() 
        if key.startswith("conditionalBlock") and isinstance(value, list)
    }

    # Tính toán lưới lỗi cho từng aiFunctionBody
    all_error_grids = []
    print("\n--- Tính toán lưới lỗi cho từng AI Function ---")
    for i, ai_body in enumerate(ai_function_bodies):
        try:
            error_grid_from_js = _execute_js_ai_function_grid_level(
                ai_body,
                raw_slides_np.tolist(),
                target_grid_np.tolist(),
                user_gradient_flows_np.tolist(),
                {name: arr.tolist() for name, arr in conditional_blocks_np_map.items()},
                session_dimensions['channels'],
                session_dimensions['height'],
                session_dimensions['width'],
                base_session_data['comparisonMode'],
                base_session_data['toleranceEpsilon']
            )
            all_error_grids.append(np.array(error_grid_from_js))
            print(f"✅ Đã tính toán lưới lỗi cho AI Function #{i+1}.")
        except Exception as e:
            accumulated_errors.append(f"Lỗi khi thực thi hàm AI #{i+1} cho toàn bộ lưới: {e}")
            all_error_grids.append(np.zeros((session_dimensions['channels'], session_dimensions['height'], session_dimensions['width']))) # Thêm lưới rỗng để tránh lỗi

    # Tìm Hotspot dựa trên tích của các lưới lỗi
    print(f"\n--- Tìm kiếm Hotspot Lỗi {HOTSPOT_SIZE}x{HOTSPOT_SIZE} dựa trên tích của các lưới lỗi ---")
    if session_dimensions['height'] >= HOTSPOT_SIZE and session_dimensions['width'] >= HOTSPOT_SIZE:
        try:
            # Tính tích của tất cả các lưới lỗi
            product_grid_np = None
            for grid in all_error_grids:
                product_grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(1).to(_global_torch_device)
                
                kernel_tensor = torch.ones((1, 1, HOTSPOT_SIZE, HOTSPOT_SIZE), dtype=torch.float32, device=_global_torch_device)
                summed_product_windows = F.conv2d(product_grid_tensor, kernel_tensor, stride=1, padding=0)
                if product_grid_np is None:
                    product_grid_np = summed_product_windows
                product_grid_np *= summed_product_windows

            temp_error_grid_tensor = product_grid_np.reshape(session_dimensions['channels'],-1)
            max_ch_idx = torch.argmax(torch.sum(temp_error_grid_tensor,dim=-1,keepdim=False))
            product_grid_np = product_grid_np[max_ch_idx, ...]
            
            output_width_conv = session_dimensions['width'] - HOTSPOT_SIZE + 1
            # Tìm giá trị lớn nhất và vị trí của nó
            max_errors_tensor = torch.max(product_grid_np)
            max_idx = torch.argmax(product_grid_np.reshape(-1))
            product_grid_np = None
            if max_errors_tensor.item() == 0:
                target_max_abs_tensor = F.conv2d(torch.tensor(torch.from_numpy(np.abs(target_grid_np[max_ch_idx, ...])).reshape(*((all_error_grids[0])[max_ch_idx, ...]).shape), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_global_torch_device), kernel_tensor, stride=1, padding=0)
                max_idx = torch.argmax(target_max_abs_tensor)
                
            best_r_start = max_idx // output_width_conv
            best_c_start = max_idx % output_width_conv
            best_hotspot_coords = (max_ch_idx.item(), best_r_start.item(), best_c_start.item())
            
            print(f"✅ Đã tìm thấy hotspot lỗi lớn nhất tại vị trí (kenh: {best_hotspot_coords[0]}, hàng: {best_hotspot_coords[1]}, cột: {best_hotspot_coords[2]}).")
            
            # In ra tổng lỗi tại hotspot cho từng AI Function
            print("\n--- Tổng lỗi tại Hotspot đã chọn cho từng AI Function ---")
            for i, error_grid in enumerate(all_error_grids):
                ch_idx, r_start, c_start = best_hotspot_coords
                cropped_error_grid = error_grid[..., r_start:r_start+HOTSPOT_SIZE, c_start:c_start+HOTSPOT_SIZE]
                cropped_error_grid = error_grid[ch_idx, ...]
                hotspot_total_errors = np.sum(cropped_error_grid)
                print(f"Tổng lỗi tại hotspot cho AI Function #{i+1}: {hotspot_total_errors}")

            # Cắt dữ liệu phiên gốc dựa trên hotspot
            base_session_data = _crop_session_data_to_hotspot(
                base_session_data,
                best_hotspot_coords[0],
                best_hotspot_coords[1],
                best_hotspot_coords[2],
                HOTSPOT_SIZE,
                session_dimensions['channels'],
                session_dimensions['height'],
                session_dimensions['width']
            )
            
        except Exception as e:
            accumulated_errors.append(f"Lỗi khi thực hiện phân tích hotspot bằng PyTorch: {e}")
            print("Không tìm thấy hotspot lỗi (hoặc lỗi trong quá trình xử lý PyTorch). Dữ liệu không bị cắt.")
    else:
        print("\nKhông đủ tệp để so sánh hoặc kích thước lưới không đủ lớn. Bỏ qua tìm kiếm hotspot dựa trên tích.")
        base_session_data = _crop_session_data_to_hotspot(
                base_session_data,
                0,
                0,
                0,
                session_dimensions['height'],
                session_dimensions['channels'],
                session_dimensions['height'],
                session_dimensions['width']
            )

    # Ghi tệp đầu ra
    if not accumulated_errors:
        print("\n--- Kết quả Hoàn tất ---")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(base_session_data, f, indent=4)
            print(f"Đã chuyển đổi thành công. Dữ liệu phiên được lưu vào: {output_path}")
        except IOError as e:
            raise IOError(f"Lỗi: Không thể ghi tệp đầu ra tại đường dẫn: {output_path}. "
                          f"Vui lòng kiểm tra quyền truy cập thư mục. Chi tiết: {e}")
    else:
        print("\n--- Tóm tắt Lỗi Tổng thể ---")
        for err_msg in accumulated_errors:
            print(f"❌ {err_msg}")
        print("\nKhông thể tạo tệp đầu ra do có lỗi trong quá trình xử lý hoặc xác thực.")
        sys.exit(1)
        
# --- Hàm main (ĐÃ THAY ĐỔI) ---
def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi các tệp JSON dữ liệu phiên, hợp nhất các hàm AI, tìm hotspot lỗi chung và cắt dữ liệu xuống khu vực đó.")
    parser.add_argument("-o", "--output_file", required=True, help="Đường dẫn đến tệp JSON đầu ra để lưu dữ liệu phiên đã chuyển đổi hoặc cắt.")
    parser.add_argument("input_files", nargs='+', help="Đường dẫn đến các tệp JSON đầu vào.")
    
    args = parser.parse_args()

    try:
        process_multiple_sessions(args.input_files, args.output_file)
    except Exception as e:
        print(f"\n--- Lỗi trong quá trình xử lý ---")
        print(e)
        print("\nChương trình đã thoát do lỗi.")
        sys.exit(1)

if __name__ == "__main__":
    main()