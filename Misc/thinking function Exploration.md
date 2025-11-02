# Folding Collabsable Thinking function

## test 1:


[<Start Thinking>] 

```html/js/css
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
