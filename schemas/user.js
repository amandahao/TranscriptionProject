var mongoose = require('mongoose');

var model = mongoose.model('user', new mongoose.Schema({
	firstName: {type: String}
	, lastName: {type: String}
	, userName: {type: String, unique: true}
	, emailAddress: {type: String, unique: true}
	, password: {type: String}
	, salt: {type: String}
}, {
	collection: 'user', autoCreate: false
}));

exports.getModel = function() {
	return model;
}
