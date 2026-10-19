
fetch('./header.html')
  .then(response => {
    if (!response.ok) {
      throw new Error('Could not load header.html');
    }
    return response.text();
  })
  .then(html => {
    document.querySelector('#header-placeholder').innerHTML = html;
  })
  .catch(console.error);
