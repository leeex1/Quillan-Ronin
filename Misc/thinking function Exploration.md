# Folding Collabsable Thinking function

## test 1:


[<Start Thinking>] 

```html
<div class="collapsible">
  <button class="collapsible-btn">Click to expand/collapse</button>
  <div class="collapsible-content">
    This is the "thinking" content. Put your reasoning or calculations here.
  </div>
</div>

<style>
.collapsible-content {
  display: none;
  padding: 10px;
  border-left: 2px solid #888;
  margin-top: 5px;
  background-color: #f9f9f9;
}
.collapsible-btn {
  cursor: pointer;
  padding: 5px 10px;
  background-color: #eee;
  border: 1px solid #ccc;
  font-weight: bold;
}
</style>

<script>
document.querySelectorAll('.collapsible-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const content = btn.nextElementSibling;
    content.style.display = content.style.display === 'block' ? 'none' : 'block';
  });
});
</script>
```

[<End Thinking>]


## TEST 2:

```html
<!-- [<Start Thinking>] -->
<div class="foldable-box">
  <button class="foldable-toggle">🧠 Show/Hide Thinking</button>
  <div class="foldable-content">
    <p>
      This is the reasoning content. You can put notes, calculations,
      logs, or any detailed text here. By default, it is hidden.
    </p>
    <p>
      Add multiple paragraphs or even nested lists. Everything stays
      hidden until the user clicks the toggle.
    </p>
  </div>
</div>
<!-- [<End Thinking>] -->

<style>
/* Foldable container */
.foldable-box {
  margin: 10px 0;
  border: 1px solid #ccc;
  border-radius: 6px;
  background-color: #fefefe;
  padding: 5px;
}

/* Toggle button */
.foldable-toggle {
  display: inline-block;
  cursor: pointer;
  background-color: #eee;
  border: 1px solid #ccc;
  padding: 5px 10px;
  font-weight: bold;
  border-radius: 4px;
}

/* Content to hide/show */
.foldable-content {
  display: none; /* hidden by default */
  padding: 10px;
  margin-top: 5px;
  border-left: 3px solid #888;
  background-color: #fafafa;
  font-family: monospace;
  white-space: pre-wrap; /* preserves formatting */
}
</style>

<script>
// Universal foldable toggle for all boxes
document.addEventListener('DOMContentLoaded', () => {
  const toggles = document.querySelectorAll('.foldable-toggle');
  toggles.forEach(btn => {
    btn.addEventListener('click', () => {
      const content = btn.nextElementSibling;
      if (!content) return;
      content.style.display = content.style.display === 'block' ? 'none' : 'block';
    });
  });
});
</script>
```

## Test 3:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Reasoning Trace System</title>
    <style>
        /* Base foldable container */
        .reasoning-box {
            margin: 10px 0;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            background-color: #f8fafc;
            padding: 8px;
            transition: all 0.3s ease;
        }

        .reasoning-box:hover {
            border-color: #cbd5e1;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Toggle button */
        .reasoning-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 16px;
            font-weight: 600;
            border-radius: 6px;
            transition: all 0.3s ease;
            width: 100%;
            text-align: left;
        }

        .reasoning-toggle:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }

        .toggle-icon {
            transition: transform 0.3s ease;
        }

        .reasoning-toggle.expanded .toggle-icon {
            transform: rotate(180deg);
        }

        /* Content area with animation */
        .reasoning-content {
            max-height: 0;
            opacity: 0;
            overflow: hidden;
            padding: 0 15px;
            margin-top: 0;
            border-left: 3px solid #667eea;
            background-color: #ffffff;
            font-family: 'Monaco', 'Consolas', monospace;
            white-space: pre-wrap;
            transition: all 0.4s ease-in-out;
            transform: translateY(-10px);
        }

        .reasoning-content.expanded {
            max-height: 5000px; /* Large enough for most content */
            opacity: 1;
            padding: 15px;
            margin-top: 10px;
            transform: translateY(0);
        }

        /* Nested boxes - different colors for levels */
        .reasoning-box .reasoning-box {
            background-color: #f1f5f9;
            border-color: #d1d5db;
        }

        .reasoning-box .reasoning-box .reasoning-box {
            background-color: #e2e8f0;
            border-color: #9ca3af;
        }

        .reasoning-box .reasoning-toggle {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            font-size: 0.95em;
        }

        .reasoning-box .reasoning-box .reasoning-toggle {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            font-size: 0.9em;
        }

        /* Export button */
        .export-container {
            margin: 20px 0;
            text-align: center;
        }

        .export-btn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .export-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(255, 107, 107, 0.3);
        }

        /* Status indicators */
        .reasoning-status {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
            background: rgba(255,255,255,0.2);
        }

        .status-thinking { background: #ffeaa7; color: #2d3436; }
        .status-complete { background: #55efc4; color: #2d3436; }
        .status-error { background: #ff7675; color: white; }

        /* Final answer styling */
        .final-answer {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #00b894;
            margin: 20px 0;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="export-container">
        <button class="export-btn" onclick="exportReasoningTraces()">
            📥 Export All Reasoning Traces
        </button>
    </div>

    <!-- Main reasoning block -->
    <div class="reasoning-box">
        <button class="reasoning-toggle" onclick="toggleReasoning(this)">
            <span class="toggle-icon">🔽</span>
            🧠 Initial Problem Analysis
            <span class="reasoning-status status-thinking">Thinking</span>
        </button>
        <div class="reasoning-content">
            <p><strong>Problem:</strong> Calculate the area of a circle with radius 5 units.</p>
            <p><strong>Formula:</strong> Area = π × r²</p>
            <p><strong>Given:</strong> radius (r) = 5, π ≈ 3.14159</p>

            <!-- Nested reasoning block -->
            <div class="reasoning-box">
                <button class="reasoning-toggle" onclick="toggleReasoning(this)">
                    <span class="toggle-icon">🔽</span>
                    📐 Step 1: Calculate r²
                    <span class="reasoning-status status-complete">Complete</span>
                </button>
                <div class="reasoning-content">
                    <p>r² = 5 × 5 = 25</p>
                    <p>This gives us the square of the radius.</p>

                    <!-- Deeply nested block -->
                    <div class="reasoning-box">
                        <button class="reasoning-toggle" onclick="toggleReasoning(this)">
                            <span class="toggle-icon">🔽</span>
                            🔍 Verification: Check calculation
                        </button>
                        <div class="reasoning-content">
                            <p>5 × 5 = 25 ✓</p>
                            <p>Alternative calculation: 5² = 25</p>
                            <p>Both methods confirm the result.</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Another nested block -->
            <div class="reasoning-box">
                <button class="reasoning-toggle" onclick="toggleReasoning(this)">
                    <span class="toggle-icon">🔽</span>
                    🔢 Step 2: Multiply by π
                    <span class="reasoning-status status-complete">Complete</span>
                </button>
                <div class="reasoning-content">
                    <p>Area = π × 25</p>
                    <p>Using π ≈ 3.14159:</p>
                    <p>3.14159 × 25 = 78.53975</p>
                </div>
            </div>

            <!-- Final calculation block -->
            <div class="reasoning-box">
                <button class="reasoning-toggle" onclick="toggleReasoning(this)">
                    <span class="toggle-icon">🔽</span>
                    ✅ Step 3: Final Calculation
                </button>
                <div class="reasoning-content">
                    <p>Area = π × r² = 3.14159 × 25</p>
                    <p><strong>Final Result:</strong> 78.53975 square units</p>
                    <p>Rounded to 2 decimal places: 78.54 square units</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Another main reasoning block -->
    <div class="reasoning-box">
        <button class="reasoning-toggle" onclick="toggleReasoning(this)">
            <span class="toggle-icon">🔽</span>
            🤔 Alternative Approach: Using exact π
        </button>
        <div class="reasoning-content">
            <p>We can also express the answer exactly:</p>
            <p>Area = π × 5² = 25π</p>
            <p>This is the exact mathematical answer.</p>
            <p>Numerical approximation depends on the required precision.</p>
        </div>
    </div>

    <!-- Final answer -->
    <div class="final-answer">
        <strong>🎯 Final Answer:</strong> The area of the circle is approximately 78.54 square units (or exactly 25π).
    </div>

    <script>
        // Toggle reasoning content with animation
        function toggleReasoning(button) {
            const content = button.nextElementSibling;
            const isExpanding = !content.classList.contains('expanded');
            
            // Toggle button state
            button.classList.toggle('expanded');
            
            // Toggle content with animation
            content.classList.toggle('expanded');
            
            // Update status indicator if present
            const status = button.querySelector('.reasoning-status');
            if (status && isExpanding) {
                status.textContent = "Expanded";
                status.className = "reasoning-status status-complete";
            }
        }

        // Export all reasoning traces as a text file
        function exportReasoningTraces() {
            let exportText = "REASONING TRACE EXPORT\n";
            exportText += "Generated on: " + new Date().toLocaleString() + "\n";
            exportText += "=".repeat(50) + "\n\n";
            
            // Collect all reasoning content
            const reasoningBoxes = document.querySelectorAll('.reasoning-box');
            
            reasoningBoxes.forEach((box, index) => {
                const toggle = box.querySelector('.reasoning-toggle');
                const content = box.querySelector('.reasoning-content');
                
                if (toggle && content) {
                    // Get header text without icons
                    const headerText = toggle.textContent.replace(/[🔽🧠📐🔍🔢✅🤔🎯]/g, '').trim();
                    
                    // Get content text
                    const contentText = content.textContent.trim();
                    
                    // Determine nesting level
                    let level = 0;
                    let parent = box.parentElement;
                    while (parent) {
                        if (parent.classList.contains('reasoning-content')) {
                            level++;
                        }
                        parent = parent.parentElement;
                    }
                    
                    const indent = '  '.repeat(level);
                    exportText += `${indent}${headerText}\n`;
                    exportText += `${indent}${'-'.repeat(headerText.length)}\n`;
                    exportText += `${indent}${contentText}\n\n`;
                }
            });
            
            // Add final answer
            const finalAnswer = document.querySelector('.final-answer');
            if (finalAnswer) {
                exportText += "FINAL ANSWER\n";
                exportText += "=".repeat(12) + "\n";
                exportText += finalAnswer.textContent.trim() + "\n\n";
            }
            
            exportText += "=".repeat(50) + "\n";
            exportText += "End of reasoning trace export";
            
            // Create and download file
            const blob = new Blob([exportText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `reasoning-trace-${new Date().toISOString().split('T')[0]}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        // Auto-expand first level by default
        document.addEventListener('DOMContentLoaded', function() {
            const firstLevelToggles = document.querySelectorAll('.reasoning-box > .reasoning-toggle');
            firstLevelToggles.forEach(toggle => {
                setTimeout(() => {
                    toggle.click();
                }, 300);
            });
        });

        // Smooth scroll to expanded content
        function smoothScrollToElement(element) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    </script>
</body>
</html>
```
