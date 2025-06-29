// ==UserScript==
// @name         ZeptoNow Smart Price Sorter (Debounced + Draggable + Spinner + Native Style)
// @namespace    http://tampermonkey.net/
// @version      3.0
// @description  Auto-sort ZeptoNow products by price with spinner, draggable toggle button, and ZeptoNow styling without freezing browser.
// @author       You
// @match        https://www.zeptonow.com/*
// @grant        none
// ==/UserScript==

(function () {
    'use strict';

    let productContainer = null;
    let ascending = true;
    let debounceTimer = null;

    const debug = (msg) => console.log(`[🛒 ZeptoSort] ${msg}`);

    // Find the container that holds all product cards
    function findContainer() {
        const nameEl = document.querySelector('h5[data-testid="product-card-name"]');
        if (nameEl) {
            productContainer = nameEl.closest('div[class*="grid"]') || nameEl.parentElement.parentElement;
            debug(`Product container found.`);
        } else {
            debug('❗️ Product container not found yet. Retrying...');
            setTimeout(findContainer, 1000);
        }
    }

    // Extract the price from each product card
    function extractPrice(productCard) {
        const priceEl = productCard.querySelector('p[class~="text-[20px]"][class~="font-[700]"]');
        if (priceEl) {
            const priceText = priceEl.innerText.trim().replace(/[^\d]/g, '');
            return parseFloat(priceText);
        } else {
            debug('❗️ Price element not found in a product card.');
            return Infinity;
        }
    }

    // Show loading spinner inside the sort button
    function showSpinner() {
        const button = window.zeptoSortButton;
        if (!button) return;
        const spinner = button.querySelector('#zepto-sort-spinner');
        if (spinner) spinner.style.display = 'inline-block';
    }

    // Hide loading spinner inside the sort button
    function hideSpinner() {
        const button = window.zeptoSortButton;
        if (!button) return;
        const spinner = button.querySelector('#zepto-sort-spinner');
        if (spinner) spinner.style.display = 'none';
    }

    // Sort product cards by price
    function sortProducts() {
        if (!productContainer) return debug('❗️ Product container is missing.');

        showSpinner();

        setTimeout(() => { // Slight delay to show spinner effect
            const items = Array.from(productContainer.children);
            items.sort((a, b) => {
                return ascending ? extractPrice(a) - extractPrice(b) : extractPrice(b) - extractPrice(a);
            });
            items.forEach(item => productContainer.appendChild(item));

            hideSpinner();
            debug(`✅ Sorted ${items.length} products by ${ascending ? 'ascending' : 'descending'} price.`);
        }, 300); // 300ms just to let the spinner show visibly
    }

    // Debounce function to control sort frequency
    function debounceSort() {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            sortProducts();
        }, 500);
    }

    // Observe product container for dynamic updates
    function observeContainer() {
        const observer = new MutationObserver(mutations => {
            if (mutations.some(m => m.addedNodes.length > 0)) {
                debug('🔄 Detected new products. Debouncing sort...');
                debounceSort();
            }
        });

        observer.observe(productContainer, { childList: true, subtree: false });
        debug('👀 MutationObserver is now watching for changes with debouncing.');
    }

    // Create draggable, toggleable sort button with ZeptoNow theme
    function addSortButton() {
        const button = document.createElement('button');
        button.innerText = 'Sort: Ascending';

        // Spinner element inside the button
        const spinner = document.createElement('span');
        spinner.id = 'zepto-sort-spinner';
        spinner.style.display = 'none';
        spinner.style.marginLeft = '10px';
        spinner.style.border = '3px solid #f3f3f3';
        spinner.style.borderTop = '3px solid #fff';
        spinner.style.borderRadius = '50%';
        spinner.style.width = '16px';
        spinner.style.height = '16px';
        spinner.style.display = 'inline-block';
        spinner.style.verticalAlign = 'middle';
        spinner.style.animation = 'spin-btn 1s linear infinite';

        // Add spinner to button
        button.appendChild(spinner);

        button.style.position = 'fixed';
        button.style.top = '100px';
        button.style.right = '20px';
        button.style.padding = '12px 18px';
        button.style.backgroundColor = '#20b253'; // ZeptoNow green
        button.style.color = 'white';
        button.style.border = 'none';
        button.style.borderRadius = '8px';
        button.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
        button.style.cursor = 'grab';
        button.style.zIndex = '9999';
        button.style.fontSize = '14px';
        button.style.transition = 'background-color 0.2s ease';

        let offsetX, offsetY, isDragging = false;

        button.addEventListener('mousedown', (e) => {
            isDragging = true;
            offsetX = e.clientX - button.getBoundingClientRect().left;
            offsetY = e.clientY - button.getBoundingClientRect().top;
            button.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                button.style.left = `${e.clientX - offsetX}px`;
                button.style.top = `${e.clientY - offsetY}px`;
                button.style.right = 'auto';
            }
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                button.style.cursor = 'grab';
            }
        });

        button.addEventListener('click', () => {
            if (!isDragging) { // Ignore click when dragging
                ascending = !ascending;
                button.innerText = `Sort: ${ascending ? 'Ascending' : 'Descending'}`;
                button.appendChild(spinner); // Ensure spinner stays in button
                sortProducts();
            }
        });

        button.addEventListener('mouseenter', () => {
            button.style.backgroundColor = '#1ba84a'; // Slight hover effect
        });

        button.addEventListener('mouseleave', () => {
            button.style.backgroundColor = '#20b253';
        });

        // Add keyframes for spinner animation
        const style = document.createElement('style');
        style.innerHTML = `
            @keyframes spin-btn {
                0% { transform: rotate(0deg);}
                100% { transform: rotate(360deg);}
            }
        `;
        document.head.appendChild(style);

        document.body.appendChild(button);
        // Save reference for spinner toggling
        window.zeptoSortButton = button;
    }

    // Initialize script
    function init() {
        findContainer();

        setTimeout(() => {
            if (productContainer && productContainer.children.length > 1) {
                sortProducts();
                observeContainer();
                addSortButton();
            } else {
                debug('❗️ Still waiting for product list to load...');
                setTimeout(init, 1000);
            }
        }, 1500);
    }

    init();

})(); 