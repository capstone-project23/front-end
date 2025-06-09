import React from "react";
import { Box, Typography, Button, Grid, Paper } from "@mui/material";
import { useNavigate } from "react-router-dom";
const images = [
  "image1.jpg",
  "image2.jpg",
  "image3.jpg",
  "image4.jpg",
  "image6.jpg",
  "image5.jpg",
];
const Home = () => {
  const navigate = useNavigate();
  return (
    <Box sx={{ p: 8, textAlign: "center" }}>
      <Typography variant="h4" gutterBottom>
        당신에게 딱 맞는 스타일을 찾아보세요
      </Typography>
      <Typography variant="subtitle1" gutterBottom></Typography>

      <Box sx={{ mt: 6, mb: 10 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={() => navigate("/recommend")}
        >
          시작하기
        </Button>
      </Box>

      <Grid container spacing={2} justifyContent="center">
        {images.map((src, index) => (
          <Grid item xs={6} sm={4} key={index}>
            <Paper
              sx={{

                height: 300,
                backgroundColor: "#eee",
                backgroundSize: "cover",
                backgroundPosition: "center",
                backgroundImage: `url(/images/${src})`,
              }}
            />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Home;
