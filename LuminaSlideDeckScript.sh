function createLuminaDeck_FinalPolish() {
  // Create the presentation
  var deck = SlidesApp.create("Lumina Presentation - Final Version");
  var slides = deck.getSlides();
  
  // --- DESIGN SYSTEM ---
  var theme = {
    primary: '#008080',   // Teal
    secondary: '#2C3E50', // Dark Navy
    accent: '#3EB489',    // Mint
    bgLight: '#F4F6F6',   // Very Light Grey for cards
    textDark: '#333333',
    textLight: '#FFFFFF',
    fontHeader: 'Montserrat',
    fontBody: 'Roboto'
  };

  // 1. TITLE SLIDE
  var titleSlide = slides[0];
  setupTitleSlide(titleSlide, theme);

  // 2. PROBLEM SLIDE
  createIconSlide(deck, "The Problem: Healthcare Access Gap", [
    {icon: "💰", title: "High Cost", text: "$150-$300 per dermatologist visit"},
    {icon: "⏰", title: "Long Waits", text: "2-4 weeks for appointments"},
    {icon: "🌍", title: "Limited Access", text: "Rural areas lack specialists"},
    {icon: "❓", title: "Product Confusion", text: "$500B market with no guidance"}
  ], theme);

  // 3. SOLUTION SLIDE
  createIconSlide(deck, "Lumina: AI-Powered Skin Analysis", [
    {icon: "⚡", title: "Instant Analysis", text: "<3 second diagnosis w/ Gemini Flash"},
    {icon: "🔒", title: "Privacy-First", text: "WASM processes images in-browser"},
    {icon: "🎯", title: "Personalized", text: "Custom bundles from 1,000+ catalog"},
    {icon: "📊", title: "Track Progress", text: "Daily diary with streak tracking"}
  ], theme);

  // 4. TECH STACK (Fixed: Added Card Backgrounds)
  createCardColumnSlide(deck, "Tech Stack & Architecture", {
    left: {
      title: "Frontend",
      items: ["React 19 + Vite 7", "TypeScript", "TailwindCSS 4", "WASM (Background Removal)", "AWS Amplify (Auth)"]
    },
    right: {
      title: "Backend & AI",
      items: ["FastAPI (Python)", "Google Gemini 2.5 Flash", "Google Vision API", "AWS S3 (Storage)", "Render (Hosting)"]
    }
  }, theme);

  // 5. PRIVACY (Comparison)
  createComparisonSlide(deck, "Privacy-First Architecture", {
    traditional: {
      title: "❌ Traditional Apps",
      steps: ["• Upload full photo (2-5MB)", "• Server removes background", "• Server stores original", "⚠️ Risk: Full image on cloud"]
    },
    lumina: {
      title: "✅ Lumina Approach",
      steps: ["• WASM removes bg locally", "• Upload processed (<100KB)", "• EXIF metadata stripped", "🛡️ Win: Background never sent"]
    }
  }, theme);

  // 6. PIPELINE (Fixed Alignment)
  createFlowSlide(deck, "Multimodal AI Analysis Pipeline", [
    {step: "1", title: "Capture", desc: "User uploads photo"},
    {step: "2", title: "Process", desc: "WASM cleans image"},
    {step: "3", title: "Diagnose", desc: "Gemini 2.5 analyzes"},
    {step: "4", title: "Localize", desc: "Vision API maps"},
    {step: "5", title: "Result", desc: "Product Match"}
  ], theme);

  // 7. DEMO
  createStyledSlide(deck, "Live Demo", 
    "URL: lumina-rosy.vercel.app\n\n" +
    "[ Placeholder: Insert Screenshots Here ]", theme);

  // 8. METRICS
  createMetricsSlide(deck, "Impact & Results", [
    {metric: "<3s", label: "Analysis Time", icon: "⚡"},
    {metric: "85%", label: "Bandwidth Saved", icon: "💾"},
    {metric: "6+", label: "Conditions Supported", icon: "🔍"},
    {metric: "0", label: "PII Stored", icon: "🔒"},
    {metric: "1k+", label: "Products Indexed", icon: "🧴"},
    {metric: "100%", label: "Privacy Score", icon: "🛡️"}
  ], theme);

  // 9. CHALLENGES
  createChallengeSlide(deck, "Engineering Challenges", [
    {challenge: "CORS blocking frontend requests", solution: "Explicit origin whitelist"},
    {challenge: "WASM SharedArrayBuffer blocked", solution: "COOP/COEP headers added"},
    {challenge: "CDN 404 errors on assets", solution: "Migrated to verified CDN"},
    {challenge: "Type mismatches", solution: "Unified TypeScript interfaces"}
  ], theme);

  // 10. ROADMAP (Fixed: No vertical text)
  createTimelineSlide(deck, "Future Roadmap", [
    {label: "Q1", year: "2026", item: "Vertex AI Migration", desc: "True ensemble model inference"},
    {label: "Q2", year: "2026", item: "Weekly Routines", desc: "Scheduling & push notifications"},
    {label: "Q3", year: "2026", item: "Mobile App", desc: "React Native for iOS/Android"},
    {label: "Q4", year: "2026", item: "Provider Integration", desc: "Telehealth API referrals"}
  ], theme);

  // 11. TEAM
  createStyledSlide(deck, "The Team", 
    "Akkeem Tyrell - Lead Developer\nIskandar Bagirov - Collaborator\n\nCUNY Tech Prep (CTP)", theme);
}

// ==========================================
// HELPER FUNCTIONS
// ==========================================

function setupTitleSlide(slide, theme) {
  slide.getBackground().setSolidFill(theme.secondary);
  var accent = slide.insertShape(SlidesApp.ShapeType.RECTANGLE, 0, 0, 40, 405); 
  accent.getFill().setSolidFill(theme.accent);
  accent.getBorder().setTransparent();

  var shapes = slide.getShapes();
  if (shapes.length > 0) {
    var title = shapes[0];
    title.getText().setText("Lumina");
    title.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(60).setForegroundColor(theme.textLight).setBold(true);
  }

  if (shapes.length > 1) {
    var sub = shapes[1];
    sub.getText().setText("Privacy-First AI Dermatological Analysis\nAkkeem Tyrell & Iskandar Bagirov");
    sub.getText().getTextStyle().setFontFamily(theme.fontBody).setForegroundColor(theme.accent).setFontSize(20);
  }
}

function createStyledSlide(deck, titleText, bodyText, theme) {
  var slide = deck.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_BODY);
  addHeaderBar(slide, theme);
  
  var title = slide.getShapes()[0];
  title.getText().setText(titleText);
  styleTitle(title, theme);

  var body = slide.getShapes()[1];
  body.getText().setText(bodyText);
  body.getText().getTextStyle().setFontFamily(theme.fontBody).setForegroundColor(theme.textDark);
}

// Fixed: Adds "Cards" behind the columns for better visibility
function createCardColumnSlide(deck, titleText, columns, theme) {
  var slide = deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
  addHeaderBar(slide, theme);
  styleCustomTitle(slide, titleText, theme);

  // Left Card
  var cardLeft = slide.insertShape(SlidesApp.ShapeType.ROUND_RECTANGLE, 40, 100, 310, 250);
  cardLeft.getFill().setSolidFill(theme.bgLight);
  cardLeft.getBorder().setTransparent();
  
  var leftTitle = slide.insertTextBox(columns.left.title, 60, 110, 270, 30);
  leftTitle.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(18).setBold(true).setForegroundColor(theme.primary);
  
  var leftList = slide.insertTextBox("• " + columns.left.items.join("\n• "), 60, 150, 270, 190);
  leftList.getText().getTextStyle().setFontFamily(theme.fontBody).setFontSize(12).setForegroundColor(theme.textDark);
  leftList.getText().getParagraphStyle().setLineSpacing(140);

  // Right Card
  var cardRight = slide.insertShape(SlidesApp.ShapeType.ROUND_RECTANGLE, 370, 100, 310, 250);
  cardRight.getFill().setSolidFill(theme.bgLight);
  cardRight.getBorder().setTransparent();

  var rightTitle = slide.insertTextBox(columns.right.title, 390, 110, 270, 30);
  rightTitle.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(18).setBold(true).setForegroundColor(theme.primary);
  
  var rightList = slide.insertTextBox("• " + columns.right.items.join("\n• "), 390, 150, 270, 190);
  rightList.getText().getTextStyle().setFontFamily(theme.fontBody).setFontSize(12).setForegroundColor(theme.textDark);
  rightList.getText().getParagraphStyle().setLineSpacing(140);
}

function createIconSlide(deck, titleText, items, theme) {
  var slide = deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
  addHeaderBar(slide, theme);
  styleCustomTitle(slide, titleText, theme);
  
  var startX = 60, startY = 110, boxW = 280, boxH = 90, gapX = 60, gapY = 30;
  
  for (var i = 0; i < items.length; i++) {
    var row = Math.floor(i / 2);
    var col = i % 2;
    var x = startX + col * (boxW + gapX);
    var y = startY + row * (boxH + gapY);
    
    // Icon Circle
    var circle = slide.insertShape(SlidesApp.ShapeType.ELLIPSE, x, y, 50, 50);
    circle.getFill().setSolidFill(theme.bgLight);
    circle.getBorder().setTransparent();
    
    var icon = slide.insertTextBox(items[i].icon, x, y + 5, 50, 40);
    icon.getText().getTextStyle().setFontSize(24);
    icon.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER);
    
    // Text
    var tBox = slide.insertTextBox(items[i].title, x + 60, y, 220, 25);
    tBox.getText().getTextStyle().setFontFamily(theme.fontHeader).setBold(true).setFontSize(14).setForegroundColor(theme.secondary);
    
    var dBox = slide.insertTextBox(items[i].text, x + 60, y + 25, 220, 60);
    dBox.getText().getTextStyle().setFontFamily(theme.fontBody).setFontSize(11).setForegroundColor(theme.textDark);
  }
}

function createComparisonSlide(deck, titleText, comp, theme) {
  var slide = deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
  addHeaderBar(slide, theme);
  styleCustomTitle(slide, titleText, theme);

  // Red Box
  var rBox = slide.insertShape(SlidesApp.ShapeType.RECTANGLE, 40, 100, 310, 250);
  rBox.getFill().setSolidFill('#FADBD8'); // Light Red
  rBox.getBorder().setTransparent();
  
  var rTitle = slide.insertTextBox(comp.traditional.title, 50, 110, 290, 30);
  rTitle.getText().getTextStyle().setFontFamily(theme.fontHeader).setBold(true).setForegroundColor('#C0392B');
  
  var rText = slide.insertTextBox(comp.traditional.steps.join('\n'), 50, 150, 290, 190);
  rText.getText().getTextStyle().setFontFamily(theme.fontBody).setFontSize(12).setForegroundColor(theme.textDark);
  rText.getText().getParagraphStyle().setLineSpacing(150);

  // Green Box
  var gBox = slide.insertShape(SlidesApp.ShapeType.RECTANGLE, 370, 100, 310, 250);
  gBox.getFill().setSolidFill('#D1F2EB'); // Light Green
  gBox.getBorder().setTransparent();

  var gTitle = slide.insertTextBox(comp.lumina.title, 380, 110, 290, 30);
  gTitle.getText().getTextStyle().setFontFamily(theme.fontHeader).setBold(true).setForegroundColor('#117A65');
  
  var gText = slide.insertTextBox(comp.lumina.steps.join('\n'), 380, 150, 290, 190);
  gText.getText().getTextStyle().setFontFamily(theme.fontBody).setFontSize(12).setForegroundColor(theme.textDark);
  gText.getText().getParagraphStyle().setLineSpacing(150);
}

function createFlowSlide(deck, titleText, steps, theme) {
  var slide = deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
  addHeaderBar(slide, theme);
  styleCustomTitle(slide, titleText, theme);

  var startX = 40, y = 140, w = 110, gap = 20;

  for (var i = 0; i < steps.length; i++) {
    var x = startX + i * (w + gap);

    // Number Circle
    var circ = slide.insertShape(SlidesApp.ShapeType.ELLIPSE, x + 35, y, 40, 40);
    circ.getFill().setSolidFill(theme.primary);
    circ.getBorder().setTransparent();

    var num = slide.insertTextBox(steps[i].step, x + 35, y + 5, 40, 30);
    num.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(18).setForegroundColor(theme.textLight).setBold(true);
    num.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER);

    // Text
    var title = slide.insertTextBox(steps[i].title, x, y + 50, w, 25);
    title.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(12).setBold(true).setForegroundColor(theme.secondary);
    title.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER);

    var desc = slide.insertTextBox(steps[i].desc, x, y + 75, w, 60);
    desc.getText().getTextStyle().setFontFamily(theme.fontBody).setFontSize(10).setForegroundColor(theme.textDark);
    desc.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER);

    // Arrow
    if (i < steps.length - 1) {
      var arrow = slide.insertShape(SlidesApp.ShapeType.RIGHT_ARROW, x + w - 5, y + 15, 20, 10);
      arrow.getFill().setSolidFill(theme.accent);
      arrow.getBorder().setTransparent();
    }
  }
}

function createMetricsSlide(deck, titleText, metrics, theme) {
  var slide = deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
  addHeaderBar(slide, theme);
  styleCustomTitle(slide, titleText, theme);

  var xStart = 60, yStart = 120, w = 200, h = 100;
  for (var i = 0; i < metrics.length; i++) {
    var r = Math.floor(i / 3), c = i % 3;
    var x = xStart + c * 220, y = yStart + r * 120;

    var val = slide.insertTextBox(metrics[i].metric, x, y, w, 40);
    val.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(32).setBold(true).setForegroundColor(theme.primary);
    val.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER);

    var lbl = slide.insertTextBox(metrics[i].label, x, y + 40, w, 30);
    lbl.getText().getTextStyle().setFontFamily(theme.fontBody).setFontSize(12).setForegroundColor(theme.textDark);
    lbl.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER);
  }
}

function createChallengeSlide(deck, titleText, items, theme) {
  var slide = deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
  addHeaderBar(slide, theme);
  styleCustomTitle(slide, titleText, theme);

  var y = 100;
  for (var i = 0; i < items.length; i++) {
    var rowY = y + i * 70;
    
    // Problem
    var pBox = slide.insertShape(SlidesApp.ShapeType.RECTANGLE, 40, rowY, 300, 50);
    pBox.getFill().setSolidFill('#FADBD8'); pBox.getBorder().setTransparent();
    var pTxt = slide.insertTextBox("⚠️ " + items[i].challenge, 50, rowY + 10, 280, 40);
    pTxt.getText().getTextStyle().setFontSize(11).setForegroundColor(theme.textDark);

    // Solution
    var sBox = slide.insertShape(SlidesApp.ShapeType.RECTANGLE, 390, rowY, 300, 50);
    sBox.getFill().setSolidFill('#D1F2EB'); sBox.getBorder().setTransparent();
    var sTxt = slide.insertTextBox("✅ " + items[i].solution, 400, rowY + 10, 280, 40);
    sTxt.getText().getTextStyle().setFontSize(11).setForegroundColor(theme.textDark).setBold(true);

    // Arrow
    var arr = slide.insertShape(SlidesApp.ShapeType.RIGHT_ARROW, 350, rowY + 15, 30, 20);
    arr.getFill().setSolidFill(theme.secondary); arr.getBorder().setTransparent();
  }
}

// Fixed: Separate Label and Year so text doesn't stack vertically
function createTimelineSlide(deck, titleText, items, theme) {
  var slide = deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
  addHeaderBar(slide, theme);
  styleCustomTitle(slide, titleText, theme);

  var xLine = 100, yStart = 120;
  
  // Line
  var line = slide.insertShape(SlidesApp.ShapeType.RECTANGLE, xLine, yStart, 4, 280);
  line.getFill().setSolidFill(theme.secondary); line.getBorder().setTransparent();

  for (var i = 0; i < items.length; i++) {
    var y = yStart + i * 75;

    // Dot
    var dot = slide.insertShape(SlidesApp.ShapeType.ELLIPSE, xLine - 23, y, 50, 50);
    dot.getFill().setSolidFill(theme.accent); dot.getBorder().setTransparent();

    // Q Label (Q1)
    var qLbl = slide.insertTextBox(items[i].label, xLine - 23, y + 12, 50, 30);
    qLbl.getText().getTextStyle().setFontSize(16).setBold(true).setForegroundColor(theme.textLight);
    qLbl.getText().getParagraphStyle().setParagraphAlignment(SlidesApp.ParagraphAlignment.CENTER);

    // Content
    var title = slide.insertTextBox(items[i].item + " (" + items[i].year + ")", xLine + 40, y + 5, 500, 25);
    title.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(16).setBold(true).setForegroundColor(theme.secondary);

    var desc = slide.insertTextBox(items[i].desc, xLine + 40, y + 30, 500, 30);
    desc.getText().getTextStyle().setFontFamily(theme.fontBody).setFontSize(12).setForegroundColor(theme.textDark);
  }
}

// SHARED STYLES
function addHeaderBar(slide, theme) {
  var bar = slide.insertShape(SlidesApp.ShapeType.RECTANGLE, 0, 75, 720, 5);
  bar.getFill().setSolidFill(theme.primary);
  bar.getBorder().setTransparent();
  var logo = slide.insertShape(SlidesApp.ShapeType.ELLIPSE, 660, 20, 40, 40);
  logo.getFill().setSolidFill(theme.accent); logo.getBorder().setTransparent();
}

function styleTitle(shape, theme) {
  shape.setTop(20).setLeft(30);
  shape.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(32).setForegroundColor(theme.secondary).setBold(true);
}

function styleCustomTitle(slide, text, theme) {
  var t = slide.insertTextBox(text, 30, 20, 600, 50);
  t.getText().getTextStyle().setFontFamily(theme.fontHeader).setFontSize(32).setForegroundColor(theme.secondary).setBold(true);
}