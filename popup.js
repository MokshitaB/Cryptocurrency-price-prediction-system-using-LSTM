function sendInput() {
  document.getElementById("result").innerHTML="please wait";
  const cur=document.getElementById('cur');
  const crypt=document.getElementById('crypt');
  var money=cur.selectedIndex;
  var crypto=crypt.selectedIndex;
  const url = "http://localhost:5000/hello";
  const data = { 
      'cur': money, 
      'crypt': crypto };
  
  fetch(url, {
    method: "POST",
    body: JSON.stringify(data),
    headers: {
      "Content-Type": "application/json"
    }
  })
  .then(response => response.json())
  .then(data => {
    const message = data.message;
    const result = document.getElementById("result");
    result.innerHTML="The current estimated price of "+crypt.options[crypt.selectedIndex].text
    +" is "+message+" "+cur.options[cur.selectedIndex].text
    +". Stay up-to-date with the latest price of various cyptocurrencies right here!";
  })
  .catch(error => console.error(error));
}
document.getElementById("submit").addEventListener("click", sendInput);