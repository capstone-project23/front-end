import sys
import os
import shutil
import tempfile
import paramiko
import posixpath
import httpx
import base64

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# Add the module path with a space to sys.path to allow direct import
predict_module_path = os.path.join(project_root, "toil", "personal color predict")
sys.path.append(predict_module_path)

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Any, Dict, List

from toil.utils import get_deepsearch
# Corrected imports to load directly from the modified sys.path
from skincolor import predict_personal_color
from bodyshape import predict_body_shape

# Load environment variables from .env file
# load_dotenv() # 기본 호출 대신 아래 경로 명시적 호출 사용

# Load environment variables from .env file located in the parent directory (deepshopping-main)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f".env file loaded from: {dotenv_path}") # 로딩 확인용 로그
else:
    print(f"Warning: .env file not found at {dotenv_path}. Attempting to load from default locations.")
    load_dotenv() # 기본 위치에서 다시 시도 (api 폴더 내 .env 등)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], # Allow React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Recommendation Endpoint ---
class UserInfo(BaseModel):
    name: str
    gender: str
    height: int
    weight: int
    body_shape: str
    personal_color: str
    age: int
    preference_style: str = Field(..., alias="preference style")

class RecommendationRequest(BaseModel):
    user_info: UserInfo
    query_text: str

# --- New Image Analysis Endpoint ---
@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    # Create a temporary directory to store the uploaded file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- Upload to remote Linux server ---
        try:
            remote_host = os.getenv("REMOTE_SERVER_HOST")
            remote_port = int(os.getenv("REMOTE_SERVER_PORT", 22))
            remote_user = os.getenv("REMOTE_SERVER_USER")
            remote_pass = os.getenv("REMOTE_SERVER_PASS")
            remote_path = os.getenv("REMOTE_SERVER_DEST_PATH")

            if all([remote_host, remote_user, remote_pass, remote_path]):
                ssh_client = paramiko.SSHClient()
                ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh_client.connect(hostname=remote_host, port=remote_port, username=remote_user, password=remote_pass)
                
                sftp_client = ssh_client.open_sftp()
                # Use "model.jpg" as the remote filename and posix path separator
                remote_file_path = posixpath.join(remote_path, "model.jpg")
                sftp_client.put(file_path, remote_file_path)
                
                sftp_client.close()
                ssh_client.close()
                print(f"Successfully uploaded file as model.jpg to {remote_host}:{remote_file_path}")
            else:
                print("Remote server credentials not set. Skipping upload.")
        except Exception as e:
            print(f"Failed to upload file to remote server: {e}")
        # --- End of upload logic ---

        # Define absolute paths for models required by personal color prediction
        # The script runs from the root `deepshopping-main` directory in production,
        # but for robustness, we construct paths from this file's location.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # should be deepshopping-main
        # Corrected path to the directory with a space
        personal_color_predict_dir = os.path.join(base_dir, "toil", "personal color predict")
        
        model_path = os.path.join(personal_color_predict_dir, "personal_color_model.h5")
        scaler_path = os.path.join(personal_color_predict_dir, "scaler.pkl")
        shape_predictor_path = os.path.join(personal_color_predict_dir, "shape_predictor_68_face_landmarks.dat")

        # --- Run predictions ---
        # Note: These are synchronous functions. For a high-performance server,
        # you might run them in a separate thread pool using `await asyncio.to_thread(...)`.
        # For this implementation, we call them directly.

        personal_color_result = predict_personal_color(
            image_path=file_path,
            model_path=model_path,
            scaler_path=scaler_path,
            shape_predictor_path=shape_predictor_path
        )
        body_shape_result = predict_body_shape(image_file_path=file_path)

        if not personal_color_result:
            raise HTTPException(status_code=400, detail="Could not determine personal color. Face may not be visible or clear.")
        if not body_shape_result:
            raise HTTPException(status_code=400, detail="Could not determine body shape.")

        return {
            "personal_color": personal_color_result,
            "body_shape": body_shape_result,
        }
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during image analysis: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during image analysis: {str(e)}")
    finally:
        # Clean up the temporary directory and file
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- Comment out or remove the old /api/search endpoint if not needed ---
# class SearchRequest(BaseModel):
#     query: str
#     max_results: int = 5
#
# @app.post("/api/search")
# async def search(request: SearchRequest):
#     try:
#         # This part needs to be updated if DeepSearchAgent is used differently
#         # deep_search = DeepSearchAgent() # This initialization might be incorrect
#         # results = await deep_search.search(request.query, max_results=request.max_results)
#         # return {"results": results}
#         return {"message": "Search endpoint placeholder. Please use /api/recommend for recommendations."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# --- New Recommendation Endpoint ---
@app.post("/api/recommend")
async def recommend(request: RecommendationRequest):
    try:
        main_llm_type = os.getenv("MAIN_LLM_TYPE")
        serper_api_key = os.getenv("SERPER_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not main_llm_type or not serper_api_key:
            raise HTTPException(status_code=500, detail="Missing critical environment variables (MAIN_LLM_TYPE, SERPER_API_KEY)")

        main_llm_config = {}
        if "gemini" in main_llm_type.lower():
            if not google_api_key:
                raise HTTPException(status_code=500, detail="Missing GOOGLE_API_KEY for Gemini model")
            main_llm_config = {
                "model": os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash"),
                "google_api_key": google_api_key,
                "tags": ["main_llm_recommend_api"]
            }
        elif "azure-openai" in main_llm_type.lower():
            raise NotImplementedError("Azure OpenAI configuration from env vars not fully implemented in this snippet.")
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported MAIN_LLM_TYPE: {main_llm_type}")

        agent = get_deepsearch(
            main_llm_type=main_llm_type,
            main_llm_config=main_llm_config,
            sub_llm_type=main_llm_type, 
            sub_llm_config=main_llm_config, 
            serper_api_key=serper_api_key,
            init_data={
                "user_info": request.user_info.model_dump(by_alias=True),
                "messages": []
            }
        )

        accumulated_answer = ""
        # stream_mode="values" 로 하면 LangGraph 체인의 최종 출력 스트림을 받을 수 있음
        # 하지만 여기서는 노트북과 유사하게 이벤트 기반으로 처리하여 "answer" 태그를 활용
        
        # agent.astream_events(...) 호출 결과를 await으로 받아야 함
        event_stream = await agent.astream_events(request.query_text, stream_mode=["values", "events"], version="v2")
        
        async for event in event_stream: # await으로 얻은 event_stream을 사용
            if event.get("event") == "on_chat_model_stream":
                content = event.get("data", {}).get("chunk", {}).content
                if content and "answer" in event.get("tags", []):
                    accumulated_answer += content
            # 다른 필요한 이벤트 처리 (예: on_tool_end 에서 데이터 수집 등)는 여기에 추가 가능

        # 모든 이벤트 스트림 처리 후, 최종 상태에서 정보 가져오기
        agent_state_dump = agent.dump()
        final_report_str = agent_state_dump.get("final_report", "보고서를 생성하지 못했습니다.")
        
        final_answer_text = accumulated_answer if accumulated_answer else final_report_str
        if accumulated_answer:
            final_answer_text, _ = agent.format_reference(accumulated_answer, "<sup>[[{number}]]({link})</sup>")
        
        # Get both markdown and raw JSON recommendations
        formatted_product_recommendations = agent.get_formatted_recommendations()
        product_recommendations_json = agent.get_product_recommendations()

        api_response = {
            "recommendation_text": final_answer_text,
            "final_report": final_report_str,
            "product_recommendations_markdown": formatted_product_recommendations,
            "product_recommendations": product_recommendations_json
        }

        return api_response

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error during recommendation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Pydantic Models ---
class TryOnRequest(BaseModel):
    image_url: str

@app.post("/api/try-on")
async def try_on(request: TryOnRequest):
    try:
        # 1. Download the image from the URL
        async with httpx.AsyncClient() as client:
            response = await client.get(request.image_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            image_data = response.content

        # 2. Get remote server credentials from environment
        remote_host = os.getenv("REMOTE_SERVER_HOST")
        remote_port = int(os.getenv("REMOTE_SERVER_PORT", 22))
        remote_user = os.getenv("REMOTE_SERVER_USER")
        remote_pass = os.getenv("REMOTE_SERVER_PASS")
        remote_image_path = os.getenv("REMOTE_SERVER_DEST_PATH") # Path for images
        remote_script_path = os.getenv("REMOTE_SCRIPT_PATH") # Path to catvton_workflow_runner.py
        remote_output_dir = os.getenv("REMOTE_OUTPUT_DIR") # Path where results are saved

        if not all([remote_host, remote_user, remote_pass, remote_image_path, remote_script_path, remote_output_dir]):
            raise HTTPException(status_code=500, detail="Remote server configuration is incomplete in .env file. Check REMOTE_OUTPUT_DIR.")

        # 3. Connect and upload the image, then execute the script
        with paramiko.SSHClient() as ssh_client:
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(hostname=remote_host, port=remote_port, username=remote_user, password=remote_pass)

            # Upload the image as top.jpg
            with ssh_client.open_sftp() as sftp_client:
                remote_file_path = posixpath.join(remote_image_path, "top.jpg")
                with sftp_client.file(remote_file_path, "wb") as f:
                    f.write(image_data)
                print(f"Successfully uploaded image to {remote_host}:{remote_file_path}")

            # Construct full paths for script arguments
            model_image_full_path = posixpath.join(remote_image_path, "model.jpg")
            top_image_full_path = posixpath.join(remote_image_path, "top.jpg")

            # Execute the script with full paths as arguments, without extra quotes
            command = f"python3 {remote_script_path} --model-image {model_image_full_path} --top-image {top_image_full_path}"
            print(f"Executing command: {command}")
            stdin, stdout, stderr = ssh_client.exec_command(command)
            
            exit_status = stdout.channel.recv_exit_status()
            stdout_output = stdout.read().decode('utf-8')
            stderr_output = stderr.read().decode('utf-8')

            print(f"Script stdout:\\n{stdout_output}")
            print(f"Script stderr:\\n{stderr_output}")

            if exit_status != 0:
                raise HTTPException(status_code=500, detail=f"Error executing script on remote server: {stderr_output}")

            # Find the newest file in the output directory
            find_latest_file_cmd = f"ls -t {remote_output_dir} | head -n 1"
            stdin, stdout, stderr = ssh_client.exec_command(find_latest_file_cmd)
            latest_filename = stdout.read().decode('utf-8').strip()

            if not latest_filename:
                raise HTTPException(status_code=404, detail="No output image found in the remote directory.")

            output_image_path = posixpath.join(remote_output_dir, latest_filename)
            print(f"Found latest output file: {output_image_path}")

            # Download the output image
            with ssh_client.open_sftp() as sftp_client:
                with sftp_client.file(output_image_path, "rb") as f:
                    output_image_data = f.read()

            # Encode image to base64
            encoded_image = base64.b64encode(output_image_data).decode('utf-8')
            base64_image_str = f"data:image/png;base64,{encoded_image}"

        return {"message": "Virtual try-on process completed successfully.", "output_image": base64_image_str}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")
    except Exception as e:
        print(f"Error during try-on process: {e}")
        import traceback
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during the try-on process: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)