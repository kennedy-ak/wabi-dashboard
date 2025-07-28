# Furniture Category Classification System - Implementation Guide

## Overview

This is a comprehensive furniture categorization and style classification system built with FastAPI and Streamlit. The system uses OpenAI's GPT models to classify furniture items by both category (SOFA, CHAIR, etc.) and design style (Modern, Japandi, Bohemian, etc.), providing detailed style tags and room placement recommendations.

## Architecture Overview

```
wabi-dashboard/
├── main.py                     # FastAPI application entry point
├── app.py                      # Streamlit frontend application
├── main_original.py            # Backup of original monolithic code
├── config/                     # Configuration management
├── api/                        # FastAPI application structure
└── IMPLEMENTATION_GUIDE.md     # This documentation file
```

---

## 🔧 Configuration Layer

### `config/settings.py`
**Purpose**: Centralized configuration management for the entire application.

**Key Features**:
- **API Configuration**: Title, description, version settings
- **Server Configuration**: Host and port settings
- **OpenAI Integration**: API key management
- **Processing Parameters**: Batch sizes for text vs image processing
- **CORS Settings**: Cross-origin resource sharing configuration
- **Furniture Categories**: Predefined category list

**Critical Settings**:
```python
TEXT_BATCH_SIZE = 10        # Items processed per batch (text mode)
IMAGE_BATCH_SIZE = 5        # Items processed per batch (image mode)
FURNITURE_CATEGORIES = [    # Supported furniture types
    "SOFA", "CHAIR", "BED", "TABLE", "NIGHTSTAND", 
    "STOOL", "STORAGE", "DESK", "BENCH", "OTTOMAN", 
    "LIGHTING", "DECOR", "OTHER"
]
```

### `config/__init__.py`
**Purpose**: Package initialization and settings export.
- Exposes the settings object for import across the application
- Maintains clean import structure

---

## 📊 Data Models Layer

### `api/models/schemas.py`
**Purpose**: Pydantic data models for request/response validation and serialization.

#### Key Models:

**1. CategoryClassificationRequest**
- Validates incoming API requests
- Handles both text and image processing modes
- Supports configurable product column naming

**2. CategoryResult**
- Core result model with enhanced style classification
- **New Features**: 
  - `primary_style` and `secondary_style` for hybrid classifications
  - `style_tags[]` for 3-5 descriptive characteristics
  - `placement_tags[]` for room placement recommendations
- Maintains backward compatibility with existing `category` and `confidence` fields

**3. BatchResult**
- Container for batch processing results
- Includes metadata: batch_id, timestamp, processing_type
- Error handling with optional error field

**4. HealthResponse & RootResponse**
- API health monitoring and documentation endpoints
- System status and capability reporting

### `api/models/__init__.py`
**Purpose**: Clean model imports and package initialization.

---

## 🧠 Core Intelligence Layer

### `api/utils/constants.py`
**Purpose**: Contains the comprehensive AI classification prompt with detailed style guidelines.

#### **Enhanced Classification Prompt Features**:

**1. Furniture Categories**
- Traditional 13 categories (SOFA, CHAIR, BED, etc.)
- Clear definitions and subcategories

**2. Design Style Classifications**
- **7 Major Styles**: Modern, Japandi, Bohemian, Scandinavian, Rustic, Shabby Chic, Coastal
- **Detailed Guidelines** for each style:
  - Design keywords and characteristics
  - Color palette specifications  
  - Style-specific tags (3-5 per item)
  - Placement recommendations (2-3 rooms per item)

**3. Multi-Modal Instructions**
- Supports both text and image analysis
- Confidence scoring guidelines
- JSON response format specification

**Example Style Definition**:
```
MODERN:
Design Keywords: minimal, sleek, clean lines, geometric, chrome, lacquered
Color Palette: black, white, gray, beige, navy, glass, dark wood
Style Tags: minimalist, geometric frame, sleek metal, functional design
Placement Tags: living room, modern bedroom, urban loft, entryway
```

### `api/utils/scraper.py`
**Purpose**: Ethical web scraping for image-based furniture analysis.

#### **PoliteScraper Class Features**:

**1. Rate Limiting**
- Random delays (2-30 seconds) between requests
- Request counting and timing enforcement
- Longer waits on rate limit detection (429 errors)

**2. Header Randomization**
- Multiple User-Agent strings rotation
- Randomized Accept-Language headers
- Dynamic referer assignment
- DNT (Do Not Track) randomization

**3. Image Extraction**
- Multi-source image detection:
  - Standard `<img>` tags
  - CSS background-image properties
  - JavaScript embedded images (Wayfair-specific patterns)
- Smart image filtering for product relevance
- Quality prioritization (larger, more detailed images)

**4. Ethical Scraping Practices**
- Respectful delay patterns
- Session cookie clearing
- Timeout handling
- Graceful error recovery

### `api/utils/__init__.py`
**Purpose**: Utility module exports.

---

## 🚀 Service Layer (Business Logic)

### `api/services/classifier.py`
**Purpose**: Core furniture classification and style analysis engine.

#### **FurnitureCategoryClassifier Class**:

**1. Text-Based Classification (`classify_text_batch`)**
- **Input Processing**: Extracts descriptive text from all columns
- **Batch Optimization**: Processes 10 items per batch for efficiency
- **AI Integration**: Uses GPT-4 with specialized furniture prompt
- **Response Parsing**: Handles various JSON response formats
- **Error Recovery**: Fallback mechanisms for parsing failures

**2. Image-Based Classification (`classify_image_batch`)**
- **URL Processing**: Validates and processes product URLs
- **Image Scraping**: Uses PoliteScraper for ethical image extraction
- **Image Processing**: Downloads, validates, and base64 encodes images
- **Vision Analysis**: Uses GPT-4 Vision for visual style classification
- **Quality Control**: File size and format validation

**3. Advanced Result Processing**
- **`_create_category_result()`**: Parses AI responses into structured data
- **`_create_fallback_result()`**: Handles classification failures gracefully  
- **`_create_error_result()`**: Standardized error response format
- **Type Safety**: Ensures style_tags and placement_tags are always arrays

**4. Image Handling Pipeline**
- Content-type validation (images only)
- Size limits (10MB maximum)
- Base64 encoding for API transmission
- Error logging and recovery

---

## 🌐 API Layer (FastAPI Routes)

### `api/routers/classification.py`
**Purpose**: Main classification endpoint with streaming response capability.

#### **Key Features**:

**1. Streaming Classification (`/classify-category`)**
- **Request Validation**: Comprehensive input validation
- **Batch Processing**: Configurable batch sizes by mode
- **Real-time Streaming**: Server-sent events for progress updates
- **Error Handling**: Graceful error recovery and reporting

**2. Stream Generator (`generate_classification_stream`)**
- **Progress Tracking**: Batch-by-batch progress reporting
- **Result Streaming**: Real-time result delivery as processed
- **Completion Signaling**: Clear end-of-processing notification
- **Error Isolation**: Individual batch errors don't stop processing

**3. Request Processing Flow**:
```
1. Validate request data and parameters
2. Check required columns (especially for image mode)
3. Split data into optimized batches
4. Process each batch (text or image classification)
5. Stream results in real-time
6. Send completion signal
```

### `api/routers/health.py`
**Purpose**: System monitoring and API documentation endpoints.

#### **Endpoints**:

**1. Health Check (`/health`)**
- System status monitoring
- OpenAI API key validation
- Timestamp tracking
- Service identification

**2. Root Documentation (`/`)**
- API capability overview
- Endpoint documentation
- Supported categories listing
- Version information

### `api/routers/__init__.py`
**Purpose**: Router module organization and exports.

---

## 📡 FastAPI Application

### `main.py`
**Purpose**: FastAPI application entry point with proper middleware and routing.

#### **Application Structure**:

**1. Lifespan Management**
- Startup logging and initialization
- Graceful shutdown handling
- Resource cleanup

**2. Middleware Configuration**
- **CORS Setup**: Configurable cross-origin access
- **Security Headers**: Proper header management
- **Request/Response Logging**: Comprehensive request tracking

**3. Router Integration**
- Classification routes mounting
- Health check routes mounting
- Modular route organization

**4. Development Server**
- Uvicorn integration
- Hot reload support
- Configurable host/port

---

## 🎨 Frontend Application

### `app.py`
**Purpose**: Streamlit-based user interface for the classification system.

#### **Major Components**:

**1. File Upload System**
- **Multi-Format Support**: CSV, Excel (.xlsx, .xls)
- **Automatic Format Detection**: File extension-based processing
- **Data Validation**: Comprehensive input validation
- **Error Handling**: User-friendly error messages

**2. Processing Mode Selection**
- **Text-Based Mode**: Uses product descriptions and attributes
- **Image-Based Mode**: Scrapes and analyzes product images
- **Dynamic UI**: Mode-specific instructions and validation

**3. Real-Time Processing Interface**
- **Streaming Results**: Live batch-by-batch result display
- **Progress Tracking**: Visual progress indicators
- **Batch Organization**: Organized result presentation
- **Error Reporting**: Clear error communication

**4. Enhanced Results Display**
- **Style Information**: Primary/secondary style display
- **Tag Visualization**: Style tags and placement recommendations
- **Data Export**: Full results CSV download
- **Analytics Dashboard**: Style distribution analysis

**5. Advanced Analytics**
- **Metrics Display**: Total products, confidence averages, category/style counts
- **Style Breakdown**: Visual chart of style distribution
- **Data Tables**: Sortable, searchable result tables

#### **Key Functions**:

**`format_categorization_result()`**
- Converts API responses to display format
- Handles style information formatting
- Creates user-friendly column names
- Manages missing data gracefully

**`stream_categorization_results()`**
- Manages real-time API communication
- Handles streaming response parsing
- Updates UI progressively
- Aggregates final results

**`validate_csv_data()`**
- Input data validation
- Column requirement checking
- Error message generation

**File Processing Pipeline**:
```python
1. File upload → Format detection (CSV/Excel)
2. Data reading → pandas DataFrame creation
3. Validation → Error checking and reporting
4. Processing → API streaming request
5. Display → Real-time result updates
6. Export → Final results download
```

---

## 🔄 Data Flow Architecture

### **Complete Processing Pipeline**:

```
1. USER UPLOADS FILE (CSV/Excel)
   ↓
2. STREAMLIT FRONTEND (app.py)
   - File format detection
   - Data validation
   - Processing mode selection
   ↓
3. FASTAPI BACKEND (main.py)
   - Request validation
   - CORS handling
   - Route processing
   ↓
4. CLASSIFICATION SERVICE (classifier.py)
   - Batch creation
   - AI processing (text or image)
   - Result structuring
   ↓
5. AI INTEGRATION
   - GPT-4 for text analysis
   - GPT-4 Vision for image analysis
   - Style classification prompt
   ↓
6. STREAMING RESPONSE
   - Real-time result streaming
   - Progress updates
   - Error handling
   ↓
7. FRONTEND DISPLAY
   - Live result updates
   - Style visualization
   - Data export
```

---

## 🎯 Style Classification System

### **7 Comprehensive Design Styles**:

**1. MODERN**
- Characteristics: Minimal, geometric, sleek metal frames
- Colors: Black, white, gray, navy
- Tags: minimalist, functional design, linear form
- Placement: Urban loft, modern bedroom

**2. JAPANDI** 
- Characteristics: Natural wood, low profile, zen-inspired
- Colors: Natural oak, beige, muted tones
- Tags: wood slats, clean minimalism, organic texture
- Placement: Calming bedroom, meditative corner

**3. BOHEMIAN**
- Characteristics: Layered textures, rattan, eclectic patterns
- Colors: Terracotta, mustard, forest green
- Tags: artisan-made, tribal pattern, earthy vibe
- Placement: Cozy nook, reading space

**4. SCANDINAVIAN**
- Characteristics: Light wood, simple silhouettes, Nordic charm
- Colors: White, light gray, pale wood
- Tags: airy design, pastel detail, functional form
- Placement: Studio apartment, family space

**5. RUSTIC**
- Characteristics: Reclaimed wood, farmhouse charm, exposed grain
- Colors: Warm brown, natural pine
- Tags: rough-hewn texture, vintage hardware, barn-style
- Placement: Countryside bedroom, cozy cabin

**6. SHABBY CHIC**
- Characteristics: Distressed paint, floral carvings, romantic vintage
- Colors: Pastel pink, ivory, faded mint
- Tags: whitewash, feminine touch, curved leg
- Placement: Vintage bedroom, romantic vanity

**7. COASTAL**
- Characteristics: Nautical touches, weathered wood, breezy vibe
- Colors: White, soft blue, driftwood
- Tags: rope detail, marine hardware, slatted wood
- Placement: Beach house, summer retreat

---

## 🛠 Technical Requirements

### **Dependencies**:
```python
# Backend (FastAPI)
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
openai>=1.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0

# Frontend (Streamlit)  
streamlit>=1.28.0
pandas>=2.1.0
openpyxl>=3.1.0  # For Excel support

# Shared
asyncio
logging
datetime
json
base64
```

### **Environment Variables**:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### **System Requirements**:
- Python 3.8+
- 4GB+ RAM (for AI processing)
- Internet connection (for OpenAI API and web scraping)
- 1GB+ disk space

---

## 🚀 Deployment Guide

### **Development Setup**:
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key_here"

# Start FastAPI backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit frontend (new terminal)
streamlit run app.py
```

### **Production Considerations**:
- **Rate Limiting**: Implement API rate limiting
- **Caching**: Add Redis for result caching
- **Load Balancing**: Use nginx for request distribution
- **Monitoring**: Implement logging and metrics
- **Security**: Add authentication and input sanitization
- **Scaling**: Consider containerization with Docker

---

## 🔍 Error Handling & Logging

### **Comprehensive Error Management**:

**1. Input Validation Errors**
- File format validation
- Data structure validation
- Required field checking
- URL validation for image mode

**2. Processing Errors**
- AI API failures
- Image scraping failures
- Network timeout handling
- Rate limit management

**3. Response Formatting**
- JSON parsing failures
- Missing field handling
- Type conversion errors
- Confidence score validation

**4. Logging Strategy**
- Request/response logging
- Error detail capture
- Performance metrics
- User action tracking

---

## 📈 Performance Optimization

### **Current Optimizations**:

**1. Batch Processing**
- Text mode: 10 items/batch
- Image mode: 5 items/batch
- Configurable batch sizes

**2. Streaming Responses**
- Real-time result delivery
- Reduced memory usage
- Improved user experience

**3. Efficient Image Processing**
- Image size validation (10MB limit)
- Format validation
- Smart image selection

**4. Rate-Limited Scraping**
- Ethical scraping practices
- Random delay patterns
- Respectful request patterns

### **Future Optimization Opportunities**:
- Result caching system
- Database integration
- Parallel processing
- Image caching
- Response compression

---

## 🔒 Security Considerations

### **Current Security Measures**:
- Input validation and sanitization
- File type restrictions
- Image size limits
- CORS configuration
- Error message sanitization

### **Recommended Enhancements**:
- API key rotation
- Request rate limiting
- User authentication
- Input sanitization strengthening
- Audit logging
- HTTPS enforcement

---

## 📝 API Documentation

### **Main Endpoints**:

**POST `/classify-category`**
- Purpose: Furniture classification with style analysis
- Input: JSON with data array, toggle mode, product column
- Response: Streaming JSON with batch results
- Features: Real-time processing, error handling

**GET `/health`**
- Purpose: System health monitoring
- Response: Status, timestamp, configuration check

**GET `/`**
- Purpose: API documentation and capabilities
- Response: Version info, endpoints, supported categories

---

## 🧪 Testing Strategy

### **Recommended Test Coverage**:

**1. Unit Tests**
- Model validation testing
- Classification logic testing
- Error handling verification
- Utility function testing

**2. Integration Tests**
- API endpoint testing
- Database integration testing
- External service integration
- File processing pipeline testing

**3. End-to-End Tests**
- Complete workflow testing
- UI interaction testing
- Error scenario testing
- Performance benchmarking

---

## 📋 Maintenance Guidelines

### **Regular Maintenance Tasks**:

**1. AI Model Management**
- Monitor classification accuracy
- Update prompts based on feedback
- Test new model versions
- Performance benchmarking

**2. Data Quality**
- Monitor classification confidence scores
- Review error patterns
- Update style guidelines
- Validate new furniture categories

**3. System Health**
- Log analysis and monitoring
- Performance optimization
- Security patch management
- Dependency updates

**4. User Experience**
- UI/UX improvements
- Feature usage analytics
- User feedback integration
- Documentation updates

---

This comprehensive implementation provides a robust, scalable furniture classification system with advanced style analysis capabilities. The modular architecture ensures maintainability while the streaming interface provides excellent user experience for processing large datasets.