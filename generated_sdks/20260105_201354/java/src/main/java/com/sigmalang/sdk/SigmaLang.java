package com.sigmalang.sdk;

import com.google.gson.Gson;
import okhttp3.*;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * SigmaLang Java SDK
 * ==================
 *
 * Enterprise-grade SDK for SigmaLang semantic compression.
 */
public class SigmaLang {
    private final OkHttpClient client;
    private final String baseUrl;
    private final String apiKey;
    private final Gson gson;

    public SigmaLang(String apiKey) {
        this(apiKey, "https://api.sigmalang.com");
    }

    public SigmaLang(String apiKey, String baseUrl) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.replaceAll("/$", "");
        this.client = new OkHttpClient();
        this.gson = new Gson();
    }

    /**
     * Compress text using SigmaLang semantic compression
     */
    public String compress(String text) throws IOException {
        return compress(text, null);
    }

    public String compress(String text, CompressOptions options) throws IOException {
        Map<String, Object> payload = new HashMap<>();
        payload.put("text", text);

        if (options != null) {
            if (options.getLevel() != null) {
                payload.put("level", options.getLevel());
            }
            payload.put("preserveFormatting", options.isPreserveFormatting());
        }

        String response = post("/compress", gson.toJson(payload));

        // Parse response
        Map<String, Object> result = gson.fromJson(response, Map.class);
        return (String) result.get("compressed");
    }

    /**
     * Decompress SigmaLang compressed data
     */
    public String decompress(String compressed) throws IOException {
        Map<String, String> payload = new HashMap<>();
        payload.put("compressed", compressed);

        String response = post("/decompress", gson.toJson(payload));

        // Parse response
        Map<String, Object> result = gson.fromJson(response, Map.class);
        return (String) result.get("text");
    }

    /**
     * Analyze text semantic structure
     */
    public AnalysisResult analyze(String text) throws IOException {
        Map<String, String> payload = new HashMap<>();
        payload.put("text", text);

        String response = post("/analyze", gson.toJson(payload));
        return gson.fromJson(response, AnalysisResult.class);
    }

    private String post(String endpoint, String jsonBody) throws IOException {
        RequestBody body = RequestBody.create(jsonBody, MediaType.parse("application/json"));

        Request request = new Request.Builder()
            .url(baseUrl + endpoint)
            .addHeader("Authorization", "Bearer " + apiKey)
            .addHeader("Content-Type", "application/json")
            .post(body)
            .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("API request failed: " + response.code() + " " + response.message());
            }

            ResponseBody responseBody = response.body();
            return responseBody != null ? responseBody.string() : "";
        }
    }

    /**
     * Compression options
     */
    public static class CompressOptions {
        private String level;
        private boolean preserveFormatting;

        public String getLevel() { return level; }
        public void setLevel(String level) { this.level = level; }

        public boolean isPreserveFormatting() { return preserveFormatting; }
        public void setPreserveFormatting(boolean preserveFormatting) { this.preserveFormatting = preserveFormatting; }
    }

    /**
     * Analysis result
     */
    public static class AnalysisResult {
        private double compressionRatio;
        private double semanticDensity;
        private double complexity;

        public double getCompressionRatio() { return compressionRatio; }
        public double getSemanticDensity() { return semanticDensity; }
        public double getComplexity() { return complexity; }
    }
}
