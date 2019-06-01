// adds a new list row that the user inputs
var startingBlacklistNumber = 2;
var blacklistItemNumber = 2; // sets the unique numbered id for each list item

function addNewBlacklist() {
  var req = new XMLHttpRequest();
  req.open('POST', '/blacklist', true);
  req.setRequestHeader('content-type', 'application/json');
  req.onreadystatechange = function(){
    if(req.readyState != 4){
      return;
    }
    console.log(req.response);

  }
  req.send(JSON.stringify({
    protocol: 'http',
    domain: location.href,
    path: null,
    subdomain: 'www',
    host: null,
    tld: null,
    parentDomain: null
  }))
  blacklistItemNumber++;

  var newBlacklistItem = document.getElementById("askForBlacklist").value;
  var td1 = document.createElement("TH");
  td1.className += "blacklistItem";
  var td2 = document.createElement("TH");
  td2.className += "deleteItem";
  var newRow = document.createElement("TR"); //add a unique numbered id to the row
  newRow.id = `${blacklistItemNumber}`;
  newRow.appendChild(td1);
  newRow.appendChild(td2);
  td1.innerHTML = `${newBlacklistItem}`; // adds the new blacklist item
  td2.innerHTML = "<i class='far fa-times-circle'></i>"; // adds the delete icon
  document.querySelector("tbody").appendChild(newRow);

  for (var i = 0; i < blacklistItemNumber - startingBlacklistNumber; i++) {

    document.querySelectorAll(".deleteItem")[i].addEventListener("click", function() {
      console.log(i);
    });
  };

}



document.getElementById("addToBlacklist").addEventListener("click", addNewBlacklist);
// the enter event listener isn't working right now.
// document.getElementById("addToBlacklist").addEventListener("keydown", function (e) {
//     if (e.key === "Enter") {
//       addNewBlacklist();
//     }
//   });

function deleteFromBlacklist() {
  // TODO: use this for the delete function: document.getElementById("myTable").deleteTHead();
  // document.getElementById("blacklistTable").deleteRow(d);
  console.log("works");
}
