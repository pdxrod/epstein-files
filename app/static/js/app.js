document.addEventListener("DOMContentLoaded", function () {
    initSearchTypeToggle();
    initKeyboardShortcuts();
});

function initSearchTypeToggle() {
    const radios = document.querySelectorAll('input[name="type"]');
    const queryInput = document.querySelector('input[name="q"]');
    if (!radios.length || !queryInput) return;

    const placeholders = {
        fulltext: "Search names, subjects, dates, keywords\u2026",
        name: "Enter a person\u2019s name (handles misspellings)\u2026",
        date: "Enter a date range using the filters below\u2026",
        category: "Enter a category slug (e.g. trafficking)\u2026",
    };

    radios.forEach(function (radio) {
        radio.addEventListener("change", function () {
            queryInput.placeholder = placeholders[this.value] || placeholders.fulltext;
        });
    });
}

function initKeyboardShortcuts() {
    document.addEventListener("keydown", function (e) {
        if (e.key === "/" && !isInputFocused()) {
            e.preventDefault();
            var searchInput = document.querySelector(".search-input-large");
            if (searchInput) searchInput.focus();
        }
    });
}

function isInputFocused() {
    var tag = document.activeElement.tagName.toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select";
}
