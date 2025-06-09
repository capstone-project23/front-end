import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import { useNavigate } from "react-router-dom";
import {
  Button,
  TextField,
  Typography,
  Container,
  Grid,
  Paper,
  Box,
  CircularProgress,
  Alert,
  Divider,
  Stack,
  MenuItem,
  Card,
  CardMedia,
  CardContent,
  CardActions,
  Modal,
} from "@mui/material";

// UserInfo 모델과 일치하는 초기 상태
const initialUserInfo = {
  name: "",
  gender: "",
  height: "", // 숫자로 변환 필요
  weight: "", // 숫자로 변환 필요
  body_shape: "",
  personal_color: "",
  age: "", // 숫자로 변환 필요
  "preference style": "", // Pydantic 모델의 alias와 일치
};

const PREFERENCE_STYLES = [
  "미니멀",
  "페미닌",
  "스트릿",
  "댄디",
  "캐주얼",
  "러블리",
  "빈티지",
];

// 초기 추천 상태
const initialRecommendationState = {
  recommendation_text: "",
  final_report: "",
  product_recommendations_markdown: "",
};

function RecommendationForm() {
  const [imagePreview, setImagePreview] = useState(null);
  const navigate = useNavigate();
  const [userInfo, setUserInfo] = useState(initialUserInfo);
  const [queryText, setQueryText] = useState("");
  const [recommendation, setRecommendation] = useState(
    initialRecommendationState
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [parsedProducts, setParsedProducts] = useState([]);
  const [tryOnLoading, setTryOnLoading] = useState(null);
  const [tryOnResultImage, setTryOnResultImage] = useState(null);
  const [isTryOnModalOpen, setIsTryOnModalOpen] = useState(false);

  // New states for image analysis
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState("");

  const handleUserInfoChange = (event) => {
    const { name, value } = event.target;
    setUserInfo((prev) => ({ ...prev, [name]: value }));
  };

  const handleQueryChange = (event) => {
    setQueryText(event.target.value);
  };

  const handleImageFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleImageAnalysis = async () => {
    if (!selectedFile) {
      setAnalysisError("먼저 이미지 파일을 선택해주세요.");
      return;
    }
    setAnalysisLoading(true);
    setAnalysisError("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://localhost:8000/api/analyze-image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      // Update form fields with analysis results
      setUserInfo((prev) => ({
        ...prev,
        personal_color: response.data.personal_color || prev.personal_color,
        body_shape: response.data.body_shape || prev.body_shape,
      }));
      alert("이미지 분석이 완료되어 퍼스널 컬러와 체형 정보가 자동으로 입력되었습니다.");
    } catch (err) {
      console.error("Error during image analysis:", err);
      if (err.response && err.response.data && err.response.data.detail) {
        setAnalysisError(
          `이미지 분석 중 오류 발생: ${err.response.data.detail}`
        );
      } else {
        setAnalysisError("이미지 분석 중 알 수 없는 오류가 발생했습니다.");
      }
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setRecommendation(initialRecommendationState);

    const formattedUserInfo = {
      ...userInfo,
      height: parseInt(userInfo.height, 10) || 0,
      weight: parseInt(userInfo.weight, 10) || 0,
      age: parseInt(userInfo.age, 10) || 0,
    };

    for (const key in formattedUserInfo) {
      if (
        formattedUserInfo[key] === "" ||
        (typeof formattedUserInfo[key] === "number" &&
          isNaN(formattedUserInfo[key]))
      ) {
        if (
          ["name", "gender", "age", "preference style"].includes(key) &&
          !formattedUserInfo[key]
        ) {
          setError(`필수 정보인 '${key}' 필드를 올바르게 입력해주세요.`);
          setLoading(false);
          return;
        }
      }
    }
    if (!queryText.trim()) {
      setError("추천 받고 싶은 내용을 입력해주세요.");
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post("http://localhost:8000/api/recommend", {
        user_info: formattedUserInfo,
        query_text: queryText,
      });

      console.log(
        "Received product_recommendations:",
        response.data.product_recommendations
      );

      setRecommendation({
        recommendation_text: response.data.recommendation_text || "",
        final_report: response.data.final_report || "",
        product_recommendations_markdown:
          response.data.product_recommendations_markdown || "",
      });

      // API 응답에서 받은 JSON 데이터를 직접 사용
      const products = response.data.product_recommendations || [];
      const allProducts = products.flatMap(p => 
        (p.products || []).map(item => ({
          name: item.title,
          price: (item.price || "0").replace(/,/g, ''),
          link: item.link,
          imageUrl: item.image
        }))
      );
      setParsedProducts(allProducts);

    } catch (err) {
      console.error("Error fetching recommendation:", err);
      if (err.response && err.response.data && err.response.data.detail) {
        const errorDetail = err.response.data.detail;
        if (errorDetail.includes("429") && errorDetail.includes("quota")) {
          setError(
            "API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해주세요. (무료 사용량은 1분에 15회로 제한됩니다.)"
          );
        } else {
          setError(`추천 생성 중 오류 발생: ${errorDetail}`);
        }
      } else {
        setError("추천을 받아오는 중 알 수 없는 오류가 발생했습니다.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleTryOn = async (imageUrl, index) => {
    setTryOnLoading(index);
    setTryOnResultImage(null);
    try {
      const response = await axios.post("http://localhost:8000/api/try-on", {
        image_url: imageUrl,
      });
      alert("가상 피팅이 완료되었습니다!");
      console.log("Try-on response:", response.data);
      if (response.data.output_image) {
        setTryOnResultImage(response.data.output_image);
        setIsTryOnModalOpen(true);
      }
    } catch (err) {
      console.error("Error during try-on:", err);
      const errorMessage =
        err.response?.data?.detail || "가상 피팅 요청 중 오류가 발생했습니다.";
      alert(`오류: ${errorMessage}`);
    } finally {
      setTryOnLoading(null);
    }
  };

  const handleCloseTryOnModal = () => {
    setIsTryOnModalOpen(false);
    setTryOnResultImage(null);
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ mt: 2 }}>
        <Button
          variant="outlined"
          color="primary"
          onClick={() => navigate("/")}
        >
          홈으로 돌아가기
        </Button>
      </Box>
      <Paper elevation={3} sx={{ padding: 3, marginTop: 4, marginBottom: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center,">
          DeepShopping 옷 추천
        </Typography>
        <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            사용자 정보
          </Typography>
          <Grid container spacing={2}>
            {Object.keys(initialUserInfo).map((key) => {
              if (key === "preference style") {
                return (
                  <Grid item xs={12} sm={6} key={key}>
                    <TextField
                      select
                      fullWidth
                      label="Preference Style"
                      name="preference style"
                      value={userInfo["preference style"]}
                      onChange={handleUserInfoChange}
                      variant="outlined"
                    >
                      {PREFERENCE_STYLES.map((style) => (
                        <MenuItem key={style} value={style}>
                          {style}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                );
              }

              return (
                <Grid item xs={12} sm={6} key={key}>
                  <TextField
                    fullWidth
                    label={key
                      .replace("_", " ")
                      .replace(/\b\w/g, (l) => l.toUpperCase())} // 'preference style' -> 'Preference Style'
                    name={key}
                    value={userInfo[key]}
                    onChange={handleUserInfoChange}
                    variant="outlined"
                    type={
                      ["height", "weight", "age"].includes(key)
                        ? "number"
                        : "text"
                    }
                    // 간단한 필수 표시 (API 스키마에 따라 조정)
                    // required={key === 'name' || key === 'gender' /* ... more required fields */}
                  />
                </Grid>
              );
            })}
            <Grid item xs={12}>
              {imagePreview && (
                <Box sx={{ mt: 2, mb: 2, display: "flex", justifyContent: "center" }}>
                  <Box
                    component="img"
                    src={imagePreview}
                    alt="Selected model"
                    sx={{
                      maxHeight: 300,
                      maxWidth: "100%",
                      borderRadius: 2,
                    }}
                  />
                </Box>
              )}
              <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1 }}>
                <Button variant="outlined" component="label" fullWidth>
                  모델 이미지 선택
                  <input
                    type="file"
                    accept="image/*"
                    hidden
                    onChange={handleImageFileChange}
                  />
                </Button>
                <Button
                  variant="contained"
                  onClick={handleImageAnalysis}
                  disabled={!selectedFile || analysisLoading}
                  fullWidth
                >
                  {analysisLoading ? (
                    <CircularProgress size={24} color="inherit" />
                  ) : (
                    "선택한 이미지로 체형/컬러 분석"
                  )}
                </Button>
              </Stack>
              {analysisError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {analysisError}
                </Alert>
              )}
            </Grid>
          </Grid>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            추천 요청
          </Typography>
          <TextField
            fullWidth
            label="어떤 옷을 추천받고 싶으신가요?"
            multiline
            rows={4}
            value={queryText}
            onChange={handleQueryChange}
            variant="outlined"
            margin="normal"
            required
          />

          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            sx={{ mt: 3, mb: 2 }}
            disabled={loading}
          >
            {loading ? (
              <CircularProgress size={24} />
            ) : (
              "나에게 맞는 옷 추천받기"
            )}
          </Button>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}

          {(recommendation.recommendation_text ||
            parsedProducts.length > 0) && (
            <Box
              sx={{ mt: 4, p: 2, border: "1px dashed grey", borderRadius: 1 }}
            >
              {recommendation.recommendation_text && (
                <Box mb={3}>
                  <Typography variant="h6">AI 추천 답변:</Typography>
                  <ReactMarkdown
                    rehypePlugins={[rehypeRaw]}
                    components={{
                      sup: ({ node, ...props }) => <sup {...props} />,
                    }}
                  >
                    {recommendation.recommendation_text}
                  </ReactMarkdown>
                </Box>
              )}

              {parsedProducts.length > 0 &&
                recommendation.recommendation_text && (
                  <Divider sx={{ my: 2 }} />
                )}

              {parsedProducts.length > 0 && (
                <Box>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="h6">실제 추천 상품 정보:</Typography>
                  <Grid container spacing={2} sx={{ mt: 1 }}>
                    {parsedProducts.map((product, index) => (
                      <Grid item xs={12} sm={6} md={4} key={index}>
                        <Card>
                          <CardMedia
                            component="img"
                            image={product.imageUrl}
                            alt={product.name}
                            sx={{ height: 250, objectFit: "contain", p: 1 }}
                          />
                          <CardContent>
                            <Typography
                              gutterBottom
                              variant="body2"
                              component="div"
                              sx={{
                                height: 60,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                display: "-webkit-box",
                                WebkitLineClamp: "3",
                                WebkitBoxOrient: "vertical",
                              }}
                            >
                              {product.name}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {new Intl.NumberFormat("ko-KR").format(
                                product.price
                              )}
                              원
                            </Typography>
                          </CardContent>
                          <CardActions sx={{ justifyContent: "center" }}>
                            <Button
                              size="small"
                              variant="contained"
                              onClick={() => handleTryOn(product.imageUrl, index)}
                              disabled={tryOnLoading === index}
                            >
                              {tryOnLoading === index ? <CircularProgress size={20} /> : "입어보기"}
                            </Button>
                          </CardActions>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}
            </Box>
          )}
        </Box>
      </Paper>

      {/* Try-on result modal */}
      <Modal
        open={isTryOnModalOpen}
        onClose={handleCloseTryOnModal}
        aria-labelledby="try-on-result-modal-title"
      >
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 400,
          bgcolor: 'background.paper',
          border: '2px solid #000',
          boxShadow: 24,
          p: 4,
          textAlign: 'center',
        }}>
          <Typography id="try-on-result-modal-title" variant="h6" component="h2">
            가상 피팅 결과
          </Typography>
          {tryOnResultImage && (
            <Box
              component="img"
              src={tryOnResultImage}
              alt="Virtual try-on result"
              sx={{
                mt: 2,
                mb: 2,
                maxHeight: 500,
                maxWidth: '100%',
                display: 'block',
                marginLeft: 'auto',
                marginRight: 'auto',
              }}
            />
          )}
          <Button onClick={handleCloseTryOnModal}>닫기</Button>
        </Box>
      </Modal>

    </Container>
  );
}

export default RecommendationForm;
