import React, { useState } from "react";
import {
  Container,
  TextField,
  Button,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Box,
  CssBaseline,
} from "@mui/material";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Home from "./components/Homef";
import RecommendationForm from "./components/RecommendationForm";

function SearchPage() {

  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post("https://57b0-220-120-112-2.ngrok-free.app/api/search", {
        query: query,
        max_results: 5,
      });
      setResults(response.data.results);
    } catch (err) {
      setError("검색 중 오류가 발생했습니다. 다시 시도해주세요.");
      console.error("Search error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (

      <Container maxWidth="md" sx={{ py: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Deep Shopping Search
        </Typography>

        <Box sx={{ display: "flex", gap: 2, mb: 4 }}>
          <TextField
            fullWidth
            label="검색어를 입력하세요"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSearch()}
          />
          <Button
            variant="contained"
            onClick={handleSearch}
            disabled={loading}
            sx={{ minWidth: 100 }}
          >
            {loading ? <CircularProgress size={24} /> : "검색"}
          </Button>
        </Box>

        {error && (
          <Typography color="error" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {results.map((result, index) => (
          <Card key={index} sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                {result.title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {result.description}
              </Typography>
              {result.url && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  <a
                    href={result.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {result.url}
                  </a>
                </Typography>
              )}
            </CardContent>
          </Card>
        ))}
      </Container>
  );
}
function App() {
  return (
    <Router>
      <CssBaseline />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/recommend" element={<RecommendationForm />} />
        <Route path="/search" element={<SearchPage />} />
      </Routes>
    </Router>
    
  );
}


export default App;
